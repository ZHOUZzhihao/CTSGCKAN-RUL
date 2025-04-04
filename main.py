
import os
import argparse
import matplotlib.pyplot as plt
from train import *
from Bayes_train import *
from model import *
from visualization import *
from torch.utils.data import DataLoader, Subset
import torchprofile
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
torch.manual_seed(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.rcParams['font.family'] = 'Times New Roman' # 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD004', help='which dataset to run, FD001-FD004')
    parser.add_argument('--modes', type=str, default='train', help='train, Bayes_train or test')
    parser.add_argument('--path', type=str,
                        default='/home/ps/code/zhouzhihao/CSTGCKAN_TEST/saved_model/model/CTSG-FD002-model',
                        help='model save/load path')
    parser.add_argument('--save_path', type=str,
                        default='/home/ps/code/zhouzhihao/CSTGCKAN_TEST/saved_model/log/',
                        help='log save path')
    parser.add_argument('--epoch', type=int, default=20, help='epoch to train')#
    parser.add_argument('--num_features', type=int, default=14, help='number of features')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')#
    parser.add_argument('--LR', type=float, default=0.0001, help='learning_rate')#
    parser.add_argument('--smooth_param', type=float, default=0.1, help='none or freq')
    parser.add_argument('--train_seq_len', type=int, default=30, help='train_seq_len')
    parser.add_argument('--test_seq_len', type=int, default=30, help='test_seq_len')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='length of patch')
    parser.add_argument('--decay_step', type=float, default=100, help='length of patch')
    parser.add_argument('--decay_ratio', type=float, default=0.5, help='length of patch')
    opt = parser.parse_args()
    print(opt)
    if opt.modes == "train":
        """
        Since the proposed method has not been published in any journal, the training program is not available at this time.
        """
        # Training(opt,i)##直接根据测试集或验证集进行选择
    elif opt.modes == "Bayes_train":
        # train_bayes_opt()##贝叶斯优化三个参数

    elif opt.modes == "test":
        # TODO:  Testing the proposed methodology
        PATH = opt.path
        print(PATH)
        group_train, y_test, group_test, X_test = data_processing(opt.dataset,opt.smooth_param)

        test_dataset = SequenceDataset(mode='test',group = group_test, y_label=y_test, sequence_train=opt.train_seq_len, patch_size=opt.train_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = Conv2DKAN()

        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.load_state_dict(torch.load(PATH))
        # criterion = torch.nn.MSELoss()
        if torch.cuda.is_available():
            model = model.to(device)

        model.eval()
        result=[]
        mse_loss=0
        print(model)
        model2 = model


        with torch.no_grad():
            test_epoch_loss = 0
            for X,y in test_loader:
                if torch.cuda.is_available():
                    X=X.cuda()
                    y=y.cuda()

                y_hat_recons = model.forward(X)

                y_hat_unscale = y_hat_recons[0]*125
                result.append(y_hat_unscale.item())

        y_test.index = y_test.index
        result = y_test.join(pd.DataFrame(result))
        result.to_csv(opt.dataset+'_result.csv')
        result['RUL'].clip(upper=125, inplace=True)
        result['RUL'].clip(lower=0, inplace=True)

        error = result.iloc[:,1]-result.iloc[:,0]
        res=0
        for value in error:
            if value < 0:
                res = res + np.exp(-value / 13) - 1
            else:
                res = res + np.exp(value / 10) - 1
        rmse =  np.sqrt(np.mean(error ** 2))
        print("testing score: %1.5f" % (res))
        print("testing rmse: %1.5f" % (rmse))

        result = result.sort_values('RUL', ascending=False)
        result['RUL'].clip(upper=125, inplace=True)
        # TODO:  visualize the testing result
        visualize(result, rmse)

        count = np.sum(error > 0)
        print("Number of elements greater than 0:", count)
        from thop import profile
        flops,params = profile(model,(X,))
        print("Total FLOPs2:", flops)
        print("Total params:", params)


        def extract_features(model, dataloader):
            features = []
            labels = []

            # 假设我们从conv2层提取特征
            def hook_fn(module, input, output):
                features.append(output.view(output.size(0), -1).cpu().detach().numpy())  # Flatten the output and detach

            # Register hook on conv2 layer
            hook = model.conv1.register_forward_hook(hook_fn)

            model.eval()
            with torch.no_grad():
                for X, y in dataloader:
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y = y.cuda()
                    _ = model(X)  # Forward pass
                    labels.append(y)

            # Remove the hook after extraction
            hook.remove()

            features = np.concatenate(features, axis=0)


            return features, labels


        from sklearn.manifold import TSNE
        def apply_tsne(features, labels,num, n_components=2):
            perplexity=min(30,features.shape[0]-1)
            tsne = TSNE(perplexity=perplexity, n_components=n_components, random_state=42)
            reduced_features = tsne.fit_transform(features)

            # 可视化降维后的数据
            plt.figure(figsize=(8, 6))
            plt.rcParams.update({'font.size': 18})
            plt.rcParams['font.weight'] = 'bold'
            scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1],c=labels , cmap='jet', s=80, alpha=0.7)
            plt.colorbar(scatter)
            plt.title("#"+str(num)+" Engine of Test FD004", fontsize=20,fontweight='bold')
            plt.xlabel("Demention 1", fontsize=20)
            plt.ylabel("Demention 2", fontsize=20)
            plt.subplots_adjust(top=0.93, bottom=0.11, right=0.97, left=0.11, hspace=0, wspace=0)
            plt.margins(0.1, 0.1)
            plt.savefig('/home/ps/code/zhouzhihao/KAN-RUL/CONVKAN/FD004-'+str(num+1)+'.jpg',dpi=300)
            plt.show()
            print(reduced_features.shape)

        # TODO:  Visualization of degradation characteristics
        group_train, y_test, group_test, X_test = data_processing(opt.dataset, opt.smooth_param,)
        for i in [49]:#ange(90,93):
            test_dataset_specific = SequenceDataset(mode='test_all_specific', group=group_test, y_label=y_test,
                                               sequence_train=opt.train_seq_len,
                                               patch_size=opt.train_seq_len, engine_num=i)
            test_loader_specific = DataLoader(test_dataset_specific, batch_size=1, shuffle=False, drop_last=False)

            features, labels = extract_features(model, test_loader_specific)
            labels = [tensor.tolist() for tensor in labels]
            apply_tsne(features, labels,num=i)


        # TODO:  Forecasting at all moments
        group_train, y_test, group_test, X_test = data_processing(opt.dataset, opt.smooth_param)

        test_dataset_all = SequenceDataset(mode='test_all', group=group_test, y_label=y_test, sequence_train=opt.train_seq_len,
                                        patch_size=opt.train_seq_len)

        test_loader_all = DataLoader(test_dataset_all, batch_size=1, shuffle=False, drop_last=False)
        result_all=[]
        y_all=[]

        with torch.no_grad():
            for X,y in test_loader_all:
                if torch.cuda.is_available():
                    X=X.cuda()
                    y=y.cuda()

                y[y>125]=125

                y_hat_recons = model.forward(X)
                y_hat_unscale = y_hat_recons[0]*125
                result_all.append(y_hat_unscale.item())
                y_all.append(y.item())

        all = np.vstack((y_all, result_all))
        print(all.shape)
        error_all = all[0, :] - all[1, :]
        rmse_all = np.sqrt(np.mean(error_all ** 2))
        print("Error_all:", rmse_all)

        lengths = X_test.apply(lambda x: len(x)).values
        # print(lengths-29)
        rul_len = lengths-29
        engine_num = [i for i, count in enumerate(rul_len, 1) for _ in range(count)]
        # print(engine_num)
        all = np.vstack((engine_num, all))
        # print(all)

        df_result_all = pd.DataFrame(all.T)
        df_result_all.to_csv(opt.dataset+'_result_all.csv')

