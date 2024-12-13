import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from models import MaskedNLLLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
import pickle as pk
import datetime
from FourierGNNmodel import MaskedKLDivLoss, DialogueGCNModel

# We use seed = 27350 for reproduction of the results reported in the paper.
seed = 27350


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('MELD_features/MELD_features_raw1.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('MELD_features/MELD_features_raw1.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_graph_model(model, loss_function, kl_loss, dataloader, cuda, modals, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    scores, vids = [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        # qmask = qmask.permute(1, 0, 2)
        if args.multi_modal:
            if args.fusion_method == 'concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf], dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf], dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf], dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf], dim=-1)
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                textf = textf

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob, e_i, e_n, e_t, e_l = model(textf, acouf, visuf, qmask, umask, lengths)

        lp_1 = log_prob1.view(-1, log_prob1.size()[1])
        lp_2 = log_prob2.view(-1, log_prob2.size()[1])
        lp_3 = log_prob3.view(-1, log_prob3.size()[1])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[1])

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[1])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[1])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[1])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[1])

        labels_ = label.view(-1)
        loss = 1 * loss_function(lp_all, labels_, umask) + \
               1 * (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(
            lp_3, labels_, umask)) + \
               1 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3,
                                                                                                          kl_p_all,
                                                                                                          umask))

        lp_ = all_prob.view(-1, all_prob.size()[1])
        preds.append(torch.argmax(lp_, 1).cpu().numpy())
        labels.append(labels_.cpu().numpy())
        losses.append(loss.item())
        masks.append(umask.view(-1).cpu().numpy())

        if train:
            loss.backward()
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    labels = np.array(labels)
    preds = np.array(preds)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, avg_fscore, masks


if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of LSTM')

    parser.add_argument('--graph-model', action='store_true', default=True,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=250, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='relation', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_residue', action='store_true', default=True,
                        help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=True,
                        help='whether to use multimodal information')

    parser.add_argument('--fusion_method', default='concat',
                        help='method to use multimodal information: concat')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=True,
                        help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=True, help='whether to use modal embedding')

    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)
    if args.av_using_lstm:
        name_ = args.fusion_method + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + 'using_lstm_' + args.Dataset
    else:
        name_ = args.fusion_method + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + '_' + args.Dataset

    if args.use_speaker:
        name_ = name_ + '_speaker'
    if args.use_modal:
        name_ = name_ + '_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 1024, 'bert': 768, 'denseface': 342, 'MELD_text': 600,
                'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = feat2dim['textCNN'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.fusion_method == 'concat':
            if modals == 'avl':
                D_m = D_audio + D_visual + D_text
            elif modals == 'av':
                D_m = D_audio + D_visual
            elif modals == 'al':
                D_m = D_audio + D_text
            elif modals == 'vl':
                D_m = D_visual + D_text
        else:
            D_m = D_text
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text

    D_g = 100
    D_p = 100
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1

    seed_everything()

    model = DialogueGCNModel(args.base_model,
                             D_m=D_m, D_e=D_e, graph_hidden_size=graph_h,
                             n_speakers=n_speakers,
                             max_seq_len=200,
                             window_past=args.windowp,
                             window_future=args.windowf,
                             n_classes=n_classes,
                             dropout=args.dropout,
                             nodal_attention=args.nodal_attention,
                             no_cuda=args.no_cuda,
                             graph_type=args.graph_type,
                             use_topic=args.use_topic,
                             alpha=args.alpha,
                             multiheads=args.multiheads,
                             graph_construct=args.graph_construct,
                             use_residue=args.use_residue,
                             D_m_v=1582,
                             D_m_a=342,
                             modals=args.modals,
                             att_type=args.fusion_method,
                             av_using_lstm=args.av_using_lstm,
                             dataset=args.Dataset,
                             use_speaker=args.use_speaker,
                             use_modal=args.use_modal)

    print('Graph NN with', args.base_model, 'as base model.')
    name = 'Graph'

    total_params = sum(p.numel() for p in model.parameters())
    print("Number of parameter: %.2fM" % (total_params / 1e6))

    if cuda:
        model.cuda()

    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1 / 0.086747,
                                          1 / 0.144406,
                                          1 / 0.227883,
                                          1 / 0.160585,
                                          1 / 0.127711,
                                          1 / 0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        if args.class_weight:
            if args.graph_model:
                loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = MaskedNLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    kl_loss = MaskedKLDivLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.1,
                                                                   batch_size=batch_size,
                                                                   num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.1,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, train_labels, train_pred, train_fscore, _ = train_or_eval_graph_model(model, loss_function,
                                                                                             kl_loss, train_loader, cuda,
                                                                                             args.modals, optimizer,
                                                                                             True)

        valid_loss, valid_acc, valid_labels, valid_pred, valid_fscore, _ = train_or_eval_graph_model(model, loss_function,
                                                                                             kl_loss, valid_loader, cuda,
                                                                                             args.modals)

        test_loss, test_acc, test_label, test_pred, test_fscore, test_mask = train_or_eval_graph_model(model,
                                                                                                           loss_function,
                                                                                                           kl_loss,
                                                                                                           test_loader,
                                                                                                           cuda,
                                                                                                           args.modals)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        all_fscore.append(test_fscore)


        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                   test_fscore, round(time.time() - start_time, 2)))
        print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
        print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))


    print('Test performance..')
    print('F-Score:', max(all_fscore))
    if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
            pk.dump({}, f)
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
        record = pk.load(f)
    key_ = name_
    if record.get(key_, False):
        record[key_].append(max(all_fscore))
    else:
        record[key_] = [max(all_fscore)]
    if record.get(key_ + 'record', False):
        record[key_ + 'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    else:
        record[key_ + 'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)]
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
        pk.dump(record, f)

    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))