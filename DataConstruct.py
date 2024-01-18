''' Data Loader class for training iteration '''
import pickle
import random

import numpy as np
import torch
from torch.autograd import Variable

import Constants


class Options(object):

    def __init__(self, data_name='twitter'):
        # data options.
        # data_name = 'twitter'
        # train file path.
        self.train_data = data_name + '/cascade.txt'
        # valid file path.
        self.valid_data = data_name + '/cascadevalid.txt'
        # test file path.
        self.test_data = data_name + '/cascadetest.txt'

        # train file path.
        self.train_data_id = data_name + '/cascade_id.txt'
        # valid file path.
        self.valid_data_id = data_name + '/cascadevalid_id.txt'
        # test file path.
        self.test_data_id = data_name + '/cascadetest_id.txt'

        self.u2idx_dict = data_name + '/u2idx.pickle'

        self.idx2u_dict = data_name + '/idx2u.pickle'

        self.ui2idx_dict = data_name + '/ui2idx.pickle'

        self.idx2ui_dict = data_name + '/idx2ui.pickle'

        self.t2idx_dict = data_name+ '/t2idx.pickle'

        self.idx2t_dict = data_name+ '/idx2t.pickle'


        # save path.
        self.save_path = ''

        self.batch_size = 16

        self.net_data = data_name + '/edges.txt'
        self.net_data_refined = data_name + '/edges_refined.txt'

        self.item_inner_net_data=data_name+"/edges_item_inner.txt"
        self.item_inner_net_data_refined=data_name+"/edges_item_inner_refined.txt"

        self.item_net_data = data_name + "/edges_item.txt"


        self.repost_net_data = data_name + "/edges_reposts.txt"
        self.repost_time_data = data_name + "/edges_reposts_time.txt"


        # For model with edge time embedding
        self.item_net_data_with_time_weight = data_name + "/edges_item_weights.txt"
        self.repost_net_data_with_time_weight = data_name + "/edges_reposts_weights.txt"

        # 1. For model with user drop 2. For model with user drop + edge time embedding
        self.item_net_data_trimmed = data_name + "/edges_item_trimmed.txt"
        self.item_net_data_with_time_weight_trimmed = data_name + "/edges_item_trimmed_weights.txt"

        self.repost_net_data_trimmed = data_name + "/edges_reposts_trimmed.txt"
        self.repost_net_data_with_time_weight_trimmed = data_name + "/edges_reposts_trimmed_weights.txt"

        self.user_repost_file=data_name+"/user_repost_list.pickle"
        self.user_hidden_net=data_name+'/edges_user_hidden.txt'

        self.embed_dim = 64


class DataConstruct(object):
    ''' For data iteration '''

    def __init__(
            self, data_name, data=0, load_dict=True, cuda=True, batch_size=32, shuffle=True, test=False,
            with_EOS=True):  # data = 0 for train, 1 for valid, 2 for test
        self.options = Options(data_name)
        self.data_name=data_name
        self.options.batch_size = batch_size
        self._u2idx = {}
        self._idx2u = []
        self.data = data
        self.test = test
        self.with_EOS = with_EOS
        if not load_dict:
            self._buildIndex()

        else:
            with open(self.options.u2idx_dict, 'rb') as handle:
                self._u2idx = pickle.load(handle)
            with open(self.options.idx2u_dict, 'rb') as handle:
                self._idx2u = pickle.load(handle)
            self.user_size = len(self._u2idx)
        self._train_cascades, train_len = self._readFromFile(self.options.train_data)
        self._valid_cascades, valid_len = self._readFromFile(self.options.valid_data)
        self._test_cascades, test_len = self._readFromFile(self.options.test_data)
        self._train_cascades_timestamp = self._readFromFileTimestamp(self.options.train_data)
        self._valid_cascades_timestamp = self._readFromFileTimestamp(self.options.valid_data)
        self._test_cascades_timestamp = self._readFromFileTimestamp(self.options.test_data)
        self._train_cascades_id = self._readFromFileId(self.options.train_data_id)
        self._valid_cascades_id = self._readFromFileId(self.options.valid_data_id)
        self._test_cascades_id = self._readFromFileId(self.options.test_data_id)

        self.train_size = len(self._train_cascades)
        self.valid_size = len(self._valid_cascades)
        self.test_size = len(self._test_cascades)
        self.cuda = cuda

        if self.data == 0:
            self._n_batch = int(np.ceil(len(self._train_cascades) / batch_size))
        elif self.data == 1:
            self._n_batch = int(np.ceil(len(self._valid_cascades) / batch_size))
        else:
            self._n_batch = int(np.ceil(len(self._test_cascades) / batch_size))

        self._batch_size = self.options.batch_size

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            random_seed_int = random.randint(0, 1000)
            random.seed(random_seed_int)
            # random.shuffle(self._train_cascades)
            # random.seed(random_seed_int)
            # random.shuffle(self._train_cascades_timestamp)
            
            num = [x for x in range(0, len(self._train_cascades))]
            random.shuffle(num)
            _train_cascade = [self._train_cascades[num[i]] for i in range(0, len(num))]
            _train_cascade_timestamp = [self._train_cascades_timestamp[num[i]] for i in range(0, len(num))]
            _train_cascade_id = [self._train_cascades_id[num[i]] for i in range(0, len(num))]

            self._train_cascades = _train_cascade
            self._train_cascades_timestamp = _train_cascade_timestamp
            self._train_cascades_id = _train_cascade_id

    def _buildIndex(self):
        # compute an index of the users that appear at least once in the training and testing cascades.
        opts = self.options

        train_user_set = set()
        valid_user_set = set()
        test_user_set = set()

        lineid = 0
        for line in open(opts.train_data):
            lineid += 1
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                train_user_set.add(user)

        for line in open(opts.valid_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                valid_user_set.add(user)

        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                test_user_set.add(user)

        user_set = train_user_set | valid_user_set | test_user_set

        pos = 0
        self._u2idx['<blank>'] = pos
        self._idx2u.append('<blank>')
        pos += 1
        self._u2idx['</s>'] = pos
        self._idx2u.append('</s>')
        pos += 1

        for user in user_set:
            self._u2idx[user] = pos
            self._idx2u.append(user)
            pos += 1
        opts.user_size = len(user_set) + 2
        self.user_size = len(user_set) + 2
        print("user_size : %d" % (opts.user_size))

        import pickle
        # with open(opts.u2idx_dict, 'wb') as handle:
        #     pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(opts.idx2u_dict, 'wb') as handle:
        #     pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.options.u2idx_dict, 'wb') as handle:
            pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.options.idx2u_dict, 'wb') as handle:
            pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self._ui2idx = self._u2idx
        self._idx2ui = self._idx2u

        train_id_set = set()
        valid_id_set = set()
        test_id_set = set()

        with open(opts.train_data_id, "r") as file:
            for line in file.readlines():
                id = line.split('\n')[0]
                train_id_set.add(id)

        with open(opts.valid_data_id, "r") as file:
            for line in file.readlines():
                id = line.strip().split('\n')[0]
                valid_id_set.add(id)

        with open(opts.test_data_id, "r") as file:
            for line in file.readlines():
                id = line.strip().split('\n')[0]
                test_id_set.add(id)

        id_set = train_id_set | valid_id_set | test_id_set

        for id in id_set:
            self._ui2idx[id] = pos
            self._idx2ui.append(id)
            pos += 1

        import pickle
        with open(opts.ui2idx_dict, 'wb') as handle:
            pickle.dump(self._ui2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(opts.idx2ui_dict, 'wb') as handle:
            pickle.dump(self._idx2ui, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _readNet(self, filename):
        adj_list = [[], [], []]
        n_edges = 0
        # add self edges
        for i in range(self.user_size):
            adj_list[0].append(i)
            adj_list[1].append(i)
            adj_list[2].append(1)
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            nodes = line.strip().split(',')
            if nodes[0] not in self._u2idx.keys() or nodes[1] not in self._u2idx.keys():
                continue
            n_edges += 1
            adj_list[0].append(self._u2idx[nodes[0]])
            adj_list[1].append(self._u2idx[nodes[1]])
            adj_list[2].append(1)  # weight
        # print('edge:', n_edges/2)
        return adj_list

    def _readNet_dict_list(self, filename):
        adj_list = {}
        # add self edges
        for i in range(self.user_size):
            adj_list.setdefault(i, [i])  # [i] or []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            nodes = line.strip().split(',')
            if nodes[0] not in self._u2idx.keys() or nodes[1] not in self._u2idx.keys():
                continue
            adj_list[self._u2idx[nodes[0]]].append(self._u2idx[nodes[1]])
            adj_list[self._u2idx[nodes[1]]].append(self._u2idx[nodes[0]])
        return adj_list

    def _load_ne(self, filename, dim):
        embed_file = open(filename, 'r')
        line = embed_file.readline().strip()
        dim = int(line.split()[1])
        embeds = np.zeros((self.user_size, dim))
        for line in embed_file.readlines():
            line = line.strip().split()
            embeds[self._u2idx[line[0]], :] = np.array(line[1:])
        return embeds

    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        total_len = 0
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            userlist = []
            chunks = line.strip().split()
            for chunk in chunks:
                # try:
                user, timestamp = chunk.split(',')
                # except:
                #     print(chunk)
                if user in self._u2idx:
                    userlist.append(self._u2idx[user])

            # if len(userlist) > 1 and len(userlist) <= 500:
            #     total_len += len(userlist)
            #     if self.with_EOS:
            #         userlist.append(Constants.EOS)
            #     t_cascades.append(userlist)
            # max_len = 200
            # if len(userlist) > max_len:
            #     userlist = userlist[:max_len]
                # timestamplist = timestamplist[:max_len]
                
            # if len(userlist) <= max_len:
            if len(userlist) >= 2 and len(userlist) <= 500:
                if self.with_EOS:
                    userlist.append(Constants.EOS)
                    # timestamplist.append(Constants.EOS)
                t_cascades.append(userlist)
                # timestamps.append(timestamplist)
                
                
        return t_cascades, total_len

    def _readFromFileTimestamp(self, filename):
        """read all cascade from training or testing files. """
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            timestamplist = []
            chunks = line.strip().split()
            for chunk in chunks:
                # try:
                user, timestamp = chunk.split(',')

                max_len = 10
                if self.data_name.find("memetracker") != -1: max_len = 12
                if len(timestamp) != max_len:
                    timestamp = timestamp.ljust(max_len, '0')

                timestamp = float(timestamp)

                # timestamp = timestamp // (60 * 60 * 24)
                # except:
                #     print(chunk)
                if user in self._u2idx:
                    timestamplist.append(timestamp)

            # if len(timestamplist) > 1 and len(timestamplist) <= 500:
            #     if self.with_EOS:
            #         timestamplist.append(Constants.EOS)
            #     t_cascades.append(timestamplist)
            
            # max_len = 200
            # if len(timestamplist) > max_len:
            #     # userlist = userlist[:max_len]
            #     timestamplist = timestamplist[:max_len]
                
            # if len(userlist) <= max_len:
            if len(timestamplist) >= 2 and len(timestamplist) <= 500:
                if self.with_EOS:
                    # userlist.append(Constants.EOS)
                    timestamplist.append(Constants.EOS)
                # t_cascades.append(userlist)
                t_cascades.append(timestamplist)

        return t_cascades

    def _readFromFileId(self, filename):
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue

            chunks = line.strip().split()
            for chunk in chunks:
                # try:
                id_ = chunk.split('\n')[0]
                t_cascades.append(id_)

        return t_cascades

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            # max_len = max(len(inst) for inst in insts)
            # inst_data = np.array([
            #     inst + [Constants.PAD] * (max_len - len(inst))
            #     for inst in insts])
            
            # max_len = 220
            # inst_data = np.array([
            #     inst + [Constants.PAD] * (max_len - len(inst))
            #     for inst in insts])
            
            m_len = 200
            max_cas = max(len(inst) for inst in insts)
            if max_cas < m_len:
                max_len = max_cas
                inst_data = np.array([
                    inst + [Constants.PAD] * (max_len - len(inst))
                    for inst in insts])
            else:
                max_len = m_len
                inst_data = np.array([
                    inst + [Constants.PAD] * (max_len - len(inst)) if len(inst) < max_len else inst[:max_len]
                    for inst in insts])

            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            if self.data == 0:
                seq_insts = self._train_cascades[start_idx:end_idx]
                seq_timestamp = self._train_cascades_timestamp[start_idx:end_idx]
                seq_id = self._train_cascades_id[start_idx:end_idx]
            elif self.data == 1:
                seq_insts = self._valid_cascades[start_idx:end_idx]
                seq_timestamp = self._valid_cascades_timestamp[start_idx:end_idx]
                seq_id = self._valid_cascades_id[start_idx:end_idx]
            else:
                seq_insts = self._test_cascades[start_idx:end_idx]
                seq_timestamp = self._test_cascades_timestamp[start_idx:end_idx]
                seq_id = self._test_cascades_id[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            # seq_id=get_all_score(seq_id)
            # seq_id=pad_to_longest_id(seq_id)

            return seq_data, seq_data_timestamp, seq_id

        else:

            if self._need_shuffle:
                random_seed_int = random.randint(0, 1000)
                random.seed(random_seed_int)
            
                num = [x for x in range(0, len(self._train_cascades))]
                random.shuffle(num)
                _train_cascade = [self._train_cascades[num[i]] for i in range(0, len(num))]
                _train_cascade_timestamp = [self._train_cascades_timestamp[num[i]] for i in range(0, len(num))]
                _train_cascade_id = [self._train_cascades_id[num[i]] for i in range(0, len(num))]

                self._train_cascades = _train_cascade
                self._train_cascades_timestamp = _train_cascade_timestamp
                self._train_cascades_id = _train_cascade_id

            self._iter_count = 0
            raise StopIteration()
