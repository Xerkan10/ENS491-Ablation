import matplotlib.style
import numpy as np
from matplotlib import pyplot as plt
from os import listdir, path
import pandas as pd
from itertools import cycle
from matplotlib.pyplot import imshow


def check_params(address,parameters):
    simulations = [name for name in listdir(address)]
    keys = [key for key in parameters.keys()]
    file ='log.txt'
    for key,data in parameters.items():
        if len(data) < len(simulations):
            for x,dir in enumerate(simulations):
                loc = path.join(address,dir,file)
                f = open(loc, mode='r')
                sim_keys = []
                for i, line in enumerate(f):
                    line = line.rstrip('\n')
                    vals = line.split(' : ')
                    if len(vals) > 1:
                        param = vals[0]
                        sim_keys.append(param)
                if len(sim_keys) < len(keys):
                    for k in keys:
                        if k not in sim_keys:
                            parameters[k].insert(x,'-')
    return parameters

def log_to_dict(address):
    parameters = {}
    for dir in listdir(address):
        if dir[:3] == 'log':
            loc = path.join(address, dir)
            f = open(loc, mode='r')
            for i, line in enumerate(f):
                line = line.rstrip('\n')
                vals = line.split(':')
                if len(vals) > 1:
                    param = vals[0]
                    if param[:4] == 'Last':
                        break
                    param = param[0:len(param) - 1]
                    val = vals[1]
                    val = val[1:len(val)]
                    try:
                        val = float(val)
                    except:
                        val = str(val)
                    parameters.update({param: val})
    return parameters

def compile_results(adress):
    result_dic = {'Test_acc':None,'acc_std':None,'Train_losses':None,'loss_std':None}
    loc = path.join(adress,'vecs')
    std_loc = path.join(adress,'std')
    for dir in listdir(loc):
        file = dir.split('.')[0]
        vec = np.load(path.join(loc,dir))
        if file != 'Clipping' and file != 'Bypassed':
            result_dic[file] = vec
    for dir in listdir(std_loc):
        file = dir.split('.')[0]
        vec = np.load(path.join(std_loc,dir))
        if file == 'Test_acc':
            result_dic['acc_std'] = vec
        elif file == 'Train_losses':
            result_dic['loss_std'] = vec
    return result_dic

def load_results(file, select,pick=None, save_csv=True,latex=False,excell=True):
    dic = {'Test_acc':[],'acc_std':[],'Train_losses':[],'loss_std':[],'sim_id':[]}
    params = None
    c = 0
    results = {'Test_acc':[],'acc_std':[],'Train_losses':[],'loss_std':[]}
    for sim in listdir(file):
        sim_path = path.join(file,sim)
        #print(sim_path)
        s_id = sim_path.split('-')[-1]
        result = compile_results(sim_path)
        dic['sim_id'].append(s_id)
        for key,vals in result.items():
            if vals is not None:
                results[key].append(vals)
                dic[key].append(vals[-1])
            else:
                dic[key].append(-1)
        if c==0:
            parms = log_to_dict(sim_path)
            params = {key:[] for key in parms}
        parms = log_to_dict(sim_path)
        for key in parms:
            if key in params:
                params[key].append(parms[key])
            else:
                params[key] = [parms[key]]
        c+=1
    params = check_params(file,params)
    params_selected = {key:params[key] for key in select}
    dic = {**dic,**params_selected}
    if pick is not None:
        selection_vec = np.ones_like(dic['Test_acc'])
        for key,values in pick.items():
            dic_vals = dic[key]
            vec = np.zeros_like(selection_vec)
            for i,val in enumerate(dic_vals):
                if val in values:
                    vec[i] = 1
            selection_vec*=vec
        selection_vec = np.array(selection_vec,dtype=bool)
        for key in dic.keys():
            dic[key] = np.asarray(dic[key])[selection_vec]
    panda = pd.DataFrame(dic)
    if save_csv:
        if latex:
            test_accs, std = dic['Test_acc'],dic['acc_std']
            latex_accs = []
            for acc, var in zip(test_accs,std):
                var = round(var,2)
                acc = round(acc,2)
                latex_accs.append('{} $\pm$ {}'.format(acc,var))
            dic['Test_acc'] =latex_accs
            panda = pd.DataFrame(dic)
            panda.to_csv('results.csv')
        else:
            panda.to_csv('results.csv')

        if excell:
            panda.to_excel('results.xlsx')
    return dic,results


def graph(key,data, legends,save_fig=None):
    linestyle =['-', '--', '-.', ':']
    colors =  ['b','g','r','c', 'm', 'y','k']
    linecycler = cycle(linestyle)
    data = data[key]
    dims = [100,100]
    legends = ['ALIE','ALIE +/-']
    for d,legend in zip(data,legends):
        x = range(1,len(d)+1)
        l = next(linecycler)
        print(d[-1])
        plt.plot(x,d, linestyle=l ,markersize=3, label=legend)
        last = d[-1]
        if last > max(dims):
            dims[1] = 10
        elif last < dims[0]:
            dims[0] = 100
    if key == "Train_losses":
        dims[0] -= dims[0] / 5
        dims[1] = 3
    plt.ylim([10,100])
    plt.xlim([0,100])
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Test accuracy',fontsize=16)
    plt.legend(prop={'size': 16},loc='lower right')
    plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    plt.yticks(np.arange(0, 101, step=20),fontsize=16)
    plt.grid(linestyle='--', linewidth=0.5)
    #plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    #plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.subplots_adjust(left=0.085,bottom=.096,right=.976,top=.988)
    if save_fig is not None:
        plt.savefig(**save_fig)
    plt.show()

def graph_2(dic,results,res_key,y_lim,mode='test'):
    new_vec = []
    for tau,val in zip(dic['tau'],dic['aggr']):
        if val == 'cc':
            val+='-{}'.format(tau)
        elif val == 'ccs':
            val = 'S-CC'
        elif val == 'fl_trust':
            val = 'FL-Trust'
        new_vec.append(val)
    new_atk = []
    for num,atk in zip(dic['traitor'],dic['attack']):
        if num == 0:
            atk = 'NoATK'
        new_atk.append(atk)

    dic['aggr'] = new_vec
    dic['attack'] = new_atk

    groups = ['Lmomentum','aggr']
    #groups = ['aggr', 'Lmomentum']
    atk_colors = {'ROP':'r','ALIE':'b','IPM':'orange'
        ,'Label flip':'m','Bit flip':'g','FedAVG':'k','Baseline':'k'
                  ,'Relocated':'m','Orthogonal':'r','ROP-S':'k',
                  'reloc-OrthoToBenign':'c','OrthoToBenign':'g'}
    atk = {'alie':'ALIE','ipm':'IPM',
           'bit_flip':'Bit flip', 'label_flip':'Label flip'
        ,'reloc':'ROP','ROP-S':'ROP-S',
           'NoATK':'Baseline'}
    new_dics = []
    uniqes = []
    numeric_results = []
    matplotlib.style.use('seaborn')
    for g in groups:
        un = np.unique(np.asarray(dic[g]))
        uniqes.append(un)
    for i in uniqes[0]:
        i_bool = dic[groups[0]] == i
        for y in uniqes[1]:
            y_list = np.asarray([y for t in range(len(dic[groups[1]]))])
            y_bool = dic[groups[1]] == y_list
            tmp_dic = dict.fromkeys(dic.keys(), [])
            for key, val in dic.items():
                tmp_dic[key] = np.asarray(val)[i_bool * y_bool]
            new_dics.append(tmp_dic)
            res = np.asarray(results[res_key])
            numeric_results.append(res[i_bool * y_bool])
    figsize = (16.5, 9) ## 3 momentum
    #figsize = (16.5, 7.5)
    fig, axs = plt.subplots(len(uniqes[0]), len(uniqes[1]), figsize=figsize)
    axs = axs.flatten()
    legends = []
    lines = []
    for i,ax in enumerate(axs):
        sub_dic = new_dics[i]
        accs = numeric_results[i]
        beta = sub_dic['Lmomentum'][0]

        agg = sub_dic['aggr'][0].upper()
        agg_ = agg.split('-')
        if agg_[0]=='CC':
           tau_val = eval(agg_[1])
           tau_val = int(tau_val) if tau_val > 0.1 else tau_val
           aggx= ' \u03C4={}'.format(tau_val)
           agg = agg_[0] + aggx
        for acc, pert in zip(accs,sub_dic['attack']):
            try:
                perturbation = atk[pert]
            except:
                perturbation = pert
            color = atk_colors[perturbation]
            linestyle = '--' if perturbation == 'Baseline' else '-'
            x = range(1, len(acc) + 1)
            ax.set_ylim(y_lim)
            ax.set_xlim(0,100)
            #ax.set_facecolor('xkcd:light grey')
            if i ==0:
                lines.append(ax.plot(x,acc,color=color,linestyle=linestyle,linewidth=1.5))
                legends.append(perturbation)
            else:
                ax.plot(x, acc, color=color,linestyle=linestyle, linewidth=1.5)
        title = '\u03B2 : {} | AGG : {}'.format(beta,agg)
        ax.set_title(title, fontweight="bold", size=18)  # Title
        if i % len(uniqes[1]):
            ax.set_yticklabels([])
        else:
            #ax.set(ylabel='Test Accuracy')
            if mode =='test':
                ax.set_ylabel('Test Accuracy', fontsize=19.0)  # Y label
            else:
                ax.set_ylabel('Train loss', fontsize=19.0)  # Y label
            ax.label_outer()
            ax.tick_params(axis='y', labelsize=16)
        if i < (len(uniqes[0]) -1) * len(uniqes[1]):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Epoch', fontsize=19)  # X label
            ax.tick_params(axis='x', labelsize=16)
            ax.label_outer()
    fig.legend(lines,     # The line objects
           labels=legends,   # The labels for each line
           loc='center right',   # Position of legend "center right" # 'upper center'
           borderaxespad=0.1,
           frameon=False,
           fontsize=14.5,
           title= 'ATK',
           title_fontsize = 14.5,
           #ncol =3
           )
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(right=0.908)
    plt.show()

def graph_z(dic,results,res_key,y_lim,mode='test'):
    new_vec = []
    for tau,val in zip(dic['tau'],dic['aggr']):
        if val == 'cc':
            val+='-{}'.format(tau)
        new_vec.append(val)
    dic['aggr'] = new_vec

    groups = ['Lmomentum','aggr']
    lambs = {1:'r',10:'m',100:'b'}
    new_dics = []
    uniqes = []
    numeric_results = []
    matplotlib.style.use('seaborn')
    for g in groups:
        un = np.unique(np.asarray(dic[g]))
        uniqes.append(un)
    for i in uniqes[0]:
        i_bool = dic[groups[0]] == i
        for y in uniqes[1]:
            y_list = np.asarray([y for t in range(len(dic[groups[1]]))])
            y_bool = dic[groups[1]] == y_list
            tmp_dic = dict.fromkeys(dic.keys(), [])
            for key, val in dic.items():
                tmp_dic[key] = np.asarray(val)[i_bool * y_bool]
            new_dics.append(tmp_dic)
            res = np.asarray(results[res_key])
            numeric_results.append(res[i_bool * y_bool])
    figsize = (16.5, 9)
    #figsize = (7.5, 6)
    fig, axs = plt.subplots(len(uniqes[0]), len(uniqes[1]), figsize=figsize)
    axs = axs.flatten()
    legends = []
    lines = []
    for i,ax in enumerate(axs):
        sub_dic = new_dics[i]
        accs = numeric_results[i]
        beta = sub_dic['Lmomentum'][0]
        agg = sub_dic['aggr'][0].upper()
        agg_ = agg.split('-')
        if agg_[0]=='CC':
           tau_val = eval(agg_[1])
           tau_val = int(tau_val) if tau_val > 0.1 else tau_val
           aggx= ' \u03C4={}'.format(tau_val)
           agg = agg_[0] + aggx
        for acc, pert in zip(accs,sub_dic['z_max']):
            l = pert
            color = lambs[l]
            x = range(1, len(acc) + 1)
            ax.set_ylim(y_lim)
            ax.set_xlim(0,100)
            #ax.set_facecolor('xkcd:light grey')
            if i ==0:
                lines.append(ax.plot(x,acc,color=color,linewidth=1.5))
                legends.append('{}'.format(int(l)))
            else:
                ax.plot(x, acc, color=color, linewidth=1.5)
        title = '\u03B2 : {} | AGG : {}'.format(beta,agg)
        ax.set_title(title, fontweight="bold", size=18)  # Title
        if i % len(uniqes[1]):
            ax.set_yticklabels([])
        else:
            #ax.set(ylabel='Test Accuracy')
            if mode =='test':
                ax.set_ylabel('Test Accuracy', fontsize=19.0)  # Y label
            else:
                ax.set_ylabel('Train loss', fontsize=19.0)  # Y label
            ax.label_outer()
            ax.tick_params(axis='y', labelsize=16)
        if i < (len(uniqes[0]) -1) * len(uniqes[1]):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Epoch', fontsize=19)  # X label
            ax.tick_params(axis='x', labelsize=16)
            ax.label_outer()

    fig.legend(lines,     # The line objects
           labels=legends,   # The labels for each line
           loc='center right',   # Position of legend "center right" # 'upper center'
           borderaxespad=0.1,
           frameon=False,
           fontsize=14.5,
           title= ('Z'),
            title_fontsize = 14.5,
            #ncol =3
           )
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(right=0.925)
    plt.show()

def graph_3(dic,results,res_key,y_lim,mode='test'):
    new_vec = []
    for tau,val in zip(dic['tau'],dic['aggr']):
        if val == 'cc':
            val='CC'.format(tau)
        elif val == 'ccs':
            val = 'S-CC'
        elif val == 'krum':
            val = 'M-Krum'
        new_vec.append(val)
    new_atk = []
    for num,atk in zip(dic['traitor'],dic['attack']):
        if num == 0:
            atk = 'NoATK'
        new_atk.append(atk)

    dic['aggr'] = new_vec
    dic['attack'] = new_atk

    groups = ['Lmomentum','aggr']
    #groups = ['aggr', 'Lmomentum']
    atk_colors = {'ROP':'r','ALIE':'b','IPM':'orange'
        ,'Label flip':'m','Bit flip':'g','FedAVG':'k','Baseline':'k'
                  ,'Relocated':'m','Orthogonal':'r','ROP-S':'k',
                  'reloc-OrthoToBenign':'c','OrthoToBenign':'g','Sparse':'c'}
    atk = {'alie':'ALIE','ipm':'IPM',
           'bit_flip':'Bit flip', 'label_flip':'Label flip'
        ,'reloc':'ROP','ROP-S':'ROP-S','sparse':'Sparse',
           'NoATK':'Baseline'}
    new_dics = []
    uniqes = []
    numeric_results = []
    matplotlib.style.use('seaborn')
    for g in groups:
        un = np.unique(np.asarray(dic[g]))
        uniqes.append(un)
    for i in uniqes[0]:
        i_bool = dic[groups[0]] == i
        for y in uniqes[1]:
            y_list = np.asarray([y for t in range(len(dic[groups[1]]))])
            y_bool = dic[groups[1]] == y_list
            tmp_dic = dict.fromkeys(dic.keys(), [])
            for key, val in dic.items():
                tmp_dic[key] = np.asarray(val)[i_bool * y_bool]
            new_dics.append(tmp_dic)
            res = np.asarray(results[res_key])
            numeric_results.append(res[i_bool * y_bool])
    figsize = (18, 6) ## 3 momentum
    #figsize = (16.5, 7.5)
    fig, axs = plt.subplots(len(uniqes[0]), len(uniqes[1]), figsize=figsize)
    axs = axs.flatten()
    legends = []
    lines = []
    for i,ax in enumerate(axs):
        sub_dic = new_dics[i]
        accs = numeric_results[i]
        beta = sub_dic['Lmomentum'][0]

        agg = sub_dic['aggr'][0].upper()
        agg_ = agg.split('-')
        for acc, pert in zip(accs,sub_dic['attack']):
            try:
                perturbation = atk[pert]
            except:
                perturbation = pert
            color = atk_colors[perturbation]
            linestyle = '--' if perturbation == 'Baseline' else '-'
            x = range(1, len(acc) + 1)
            ax.set_ylim(y_lim)
            ax.set_xlim(0,100)
            #ax.set_facecolor('xkcd:light grey')
            if i ==0:
                lines.append(ax.plot(x,acc,color=color,linestyle=linestyle,linewidth=1.5))
                legends.append(perturbation)
            else:
                ax.plot(x, acc, color=color,linestyle=linestyle, linewidth=1.5)
        title = '\u03B2 : {} | AGG : {}'.format(beta,agg)
        ax.set_title(title, fontweight="bold", size=18)  # Title
        if i % len(uniqes[1]):
            ax.set_yticklabels([])
        else:
            #ax.set(ylabel='Test Accuracy')
            if mode =='test':
                ax.set_ylabel('Test Accuracy', fontsize=19.0)  # Y label
            else:
                ax.set_ylabel('Train loss', fontsize=19.0)  # Y label
            ax.label_outer()
            ax.tick_params(axis='y', labelsize=16)
        if i < (len(uniqes[0]) -1) * len(uniqes[1]):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Epoch', fontsize=19)  # X label
            ax.tick_params(axis='x', labelsize=16)
            ax.label_outer()
    fig.legend(lines,     # The line objects
           labels=legends,   # The labels for each line
           loc='center right',   # Position of legend "center right" # 'upper center'
           borderaxespad=0.1,
           frameon=False,
           fontsize=14.5,
           title= 'ATK',
           title_fontsize = 14.5,
           #ncol =3
           )
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(right=0.908)
    plt.show()

def graph_3_dist(dic,results,res_key,y_lim,mode='test'):
    new_vec = []
    for val in dic['aggr']:
        if val == 'ccs':
            val = 'S-CC'
        elif val == 'krum':
            val = 'M-Krum'
        new_vec.append(val)
    new_atk = []
    for num,atk in zip(dic['traitor'],dic['attack']):
        if num == 0:
            atk = 'NoATK'
        new_atk.append(atk)
    dic['aggr'] = new_vec
    dic['attack'] = new_atk

    groups = ['aggr','dataset_dist']
    #groups = ['aggr', 'Lmomentum']
    atk_colors = {'ROP':'r','ALIE':'b','IPM':'orange'
        ,'Label flip':'m','Bit flip':'g','FedAVG':'k','Baseline':'k','Sparse':'c'}
    atk = {'alie':'ALIE','ipm':'IPM',
           'bit_flip':'Bit flip', 'label_flip':'Label flip'
        ,'reloc':'ROP','ROP-S':'ROP-S','sparse':'Sparse',
           'NoATK':'Baseline'}
    new_dics = []
    uniqes = []
    numeric_results = []
    matplotlib.style.use('seaborn')
    for g in groups:
        un = np.unique(np.asarray(dic[g]))
        uniqes.append(un)
    for i in uniqes[0]:
        i_bool = dic[groups[0]] == i
        for y in uniqes[1]:
            y_list = np.asarray([y for t in range(len(dic[groups[1]]))])
            y_bool = dic[groups[1]] == y_list
            tmp_dic = dict.fromkeys(dic.keys(), [])
            for key, val in dic.items():
                tmp_dic[key] = np.asarray(val)[i_bool * y_bool]
            new_dics.append(tmp_dic)
            res = np.asarray(results[res_key])
            numeric_results.append(res[i_bool * y_bool])
    print(new_dics)
    #figsize = (18, 6) ## 3 momentum
    figsize = (16.5, 7.5)
    fig, axs = plt.subplots(len(uniqes[0]), len(uniqes[1]), figsize=figsize)
    axs = axs.flatten()
    legends = []
    lines = []
    for i,ax in enumerate(axs):
        sub_dic = new_dics[i]
        accs = numeric_results[i]
        dist = sub_dic['dataset_dist'][0]

        agg = sub_dic['aggr'][0].upper()
        agg_ = agg.split('-')
        for acc, pert in zip(accs,sub_dic['attack']):
            try:
                perturbation = atk[pert]
            except:
                perturbation = pert
            color = atk_colors[perturbation]
            linestyle = '--' if perturbation == 'Baseline' else '-'
            x = range(1, len(acc) + 1)
            ax.set_ylim(y_lim)
            ax.set_xlim(0,100)
            #ax.set_facecolor('xkcd:light grey')
            if i ==0:
                lines.append(ax.plot(x,acc,color=color,linestyle=linestyle,linewidth=1.5))
                legends.append(perturbation)
            else:
                ax.plot(x, acc, color=color,linestyle=linestyle, linewidth=1.5)
        title = 'AGG : {}'.format(agg)
        ax.set_title(title, fontweight="bold", size=18)  # Title
        if i % len(uniqes[1]):
            ax.set_yticklabels([])
        else:
            #ax.set(ylabel='Test Accuracy')
            if mode =='test':
                ax.set_ylabel('Test Accuracy', fontsize=19.0)  # Y label
            else:
                ax.set_ylabel('Train loss', fontsize=19.0)  # Y label
            ax.label_outer()
            ax.tick_params(axis='y', labelsize=16)
        if i < (len(uniqes[0]) -1) * len(uniqes[1]):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Epoch', fontsize=19)  # X label
            ax.tick_params(axis='x', labelsize=16)
            ax.label_outer()
    fig.legend(lines,     # The line objects
           labels=legends,   # The labels for each line
           loc='center right',   # Position of legend "center right" # 'upper center'
           borderaxespad=0.1,
           frameon=False,
           fontsize=14.5,
           title= 'ATK',
           title_fontsize = 14.5,
           #ncol =3
           )
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(right=0.908)
    plt.show()

def graph_new(dic,results,res_key,y_lim,mode='test',reverse_row=False,baseline=None,sort_key=None):
    row = 'dataset_dist'
    new_vec = []
    row_dic = {'iid':'IID','dirichlet':'non-IID'}
    for val in dic['aggr']:
        if val == 'ccs':
            val = 'S-CC'
        elif val == 'krum':
            val = 'M-Krum'
        elif val == 'bulyan':
            val = 'Bulyan'
        elif val == 'fl_trust':
            val = 'FL-Trust'
        elif val == 'krum2':
            val = 'Krum'
        elif val == 'tm_perfect':
            val = 'TM-Oracle'
        elif val == 'tm_history':
            val = 'TM-History'
        elif val == 'tm_capped':
            val = 'CliM'
        else:
            val = val.upper()
        new_vec.append(val)
    new_atk = []

    for num, atk in zip(dic['traitor'], dic['attack']):
        if num == 0:
            atk = 'NoATK'
        new_atk.append(atk)
    dic['aggr'] = new_vec
    dic['attack'] = new_atk
    label_dic = {byz:True for byz in dic['attack']}

    atk_colors = {'ROP': 'r', 'ALIE': 'b', 'IPM': 'orange'
        , 'Label flip': 'm', 'Bit flip': 'g', 'FedAVG': 'k', 'Baseline': 'k', 'HSA': 'c', 'Sparse-R': 'k',
                  'MinMax': 'olive', 'MinSum': 'indigo', 'HMS': 'darkslategray', 'C&W': 'pink'}
    atk = {'alie': 'ALIE', 'ipm': 'IPM',
           'bit_flip': 'Bit flip', 'label_flip': 'Label flip', 'cw': 'C&W'
        , 'reloc': 'ROP', 'ROP-S': 'ROP-S', 'sparse': 'HSA',
           'NoATK': 'Baseline', 'Sparse-R': 'Sparse-R', 'minmax': 'MinMax', 'minsum': 'MinSum', 'sparse_opt5': 'HMS'}
    avail_aggr = []
    for a in dic['aggr']:
        if a not in avail_aggr:
            avail_aggr.append(a)

    avail_row = []
    for a in dic[row]:
        if a not in avail_row:
            avail_row.append(a)
    if reverse_row:
        avail_row = list(reversed(avail_row)) #### IF needed
    combined = []
    avail_aggr = sorted(avail_aggr)
    if 'S-CC' in avail_aggr:
        avail_aggr.remove('S-CC')
        avail_aggr.insert(0,'S-CC')
    if sort_key is not None:
        avail_aggr = avail_aggr[sort_key]
    for r in avail_row:
        for aggr in avail_aggr:
            combined.append('{}+{}'.format(r,aggr))
    final_dict = {d:[] for d in combined}
    for r,aggr,attack,num_val in zip(dic[row],dic['aggr'],dic['attack'],results[res_key]):
        res = [attack,num_val]
        final_dict['{}+{}'.format(r,aggr)].append(res)
    row_size = 2.5 *len(avail_row)
    figsize = (20, row_size)
    matplotlib.style.use('seaborn-v0_8')
    plt.style.use("seaborn-v0_8")
    fig, axs = plt.subplots(len(avail_row), len(avail_aggr), figsize=figsize)
    axs = axs.flatten()
    lines,legends = [], []
    count = 0
    for ax,sim in zip(axs,final_dict):
        # Set consistent tick positions for proper grid alignment across all subplots
        y_ticks = np.arange(y_lim[0], y_lim[1] + 1, 20)  # Y-ticks every 20 units
        x_ticks = np.arange(0, 101, 20)  # X-ticks every 20 units: [0, 20, 40, 60, 80, 100]
        ax.set_yticks(y_ticks)
        ax.set_xticks(x_ticks)
        
        # Add grid lines with better visibility
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.6, color='white')
        
        # Draw baseline reference line if specified (don't capture for legend)
        
        for pert,acc in (final_dict[sim]):
            try:
                perturbation = atk[pert]
            except:
                perturbation = pert
            color = atk_colors[perturbation]
            #print(perturbation,color)
            linestyle = '--' if perturbation == 'Baseline' else '-'
            x = range(1, len(acc) + 1)
            ax.set_ylim(y_lim)
            ax.set_xlim(0,100)
            #ax.set_facecolor('xkcd:light grey')
            if label_dic[pert]:
                # Skip adding 'Baseline' to the main legend if we have baseline parameter active
                if not (perturbation == 'Baseline' and baseline is not None):
                    lines.append(ax.plot(x,acc,color=color,linestyle=linestyle,linewidth=1.5))
                    legends.append(perturbation)
                else:
                    ax.plot(x, acc, color=color,linestyle=linestyle, linewidth=1.5)
                label_dic[pert] = False
            else:
                ax.plot(x, acc, color=color,linestyle=linestyle, linewidth=1.5)
        row_str, aggr_str = sim.split('+')
        title = '{}'.format(aggr_str)
        if count< len(avail_aggr):
            ax.set_title(title, fontweight="bold", size=18)  # Title
        if count % len(avail_aggr) >0 and count % len(avail_aggr) != len(avail_aggr):
            # Hide y-tick labels but keep the ticks for grid alignment
            ax.tick_params(axis='y', labelleft=False)
        else:
            #ax.set(ylabel='Test Accuracy')
            if mode =='test':
                y_label = 'Test Accuracy \n {}'.format(row_dic[row_str])
                ax.set_ylabel(y_label, fontsize=19.0)  # Y label

            else:
                ax.set_ylabel('Train loss', fontsize=19.0)  # Y label
            ax.label_outer()
            ax.tick_params(axis='y', labelsize=16)
        if count < (len(avail_row) -1) * len(avail_aggr):
            # Hide x-tick labels but keep the ticks for grid alignment
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel('Epoch', fontsize=19)  # X label
            ax.tick_params(axis='x', labelsize=14)
            ax.label_outer()
        count +=1
    
    # Create baseline legend if baseline is specified
    if baseline is not None:
        for ax in axs:
            ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    # Create main attack legend
    fig.legend(lines,  # The line objects
               labels=legends,  # The labels for each line
               loc='center right',  # Position of legend "center right" # 'upper center'
               borderaxespad=0.1,
               frameon=False,
               fontsize=14.5,
               title='ATK',
               title_fontsize=14.5,
               # ncol =3
               )
    if baseline is not None:
    # Create a proper line object for the legend to show the correct line style
        from matplotlib.lines import Line2D
        baseline_legend_line = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        baseline_legend = fig.legend([baseline_legend_line], [f'Baseline'], 
                                    loc='upper right', 
                                    bbox_to_anchor=(1.0, 0.75),
                                    borderaxespad=0.1,
                                    frameon=False,
                                    fontsize=14,
                                    title_fontsize=12)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(right=0.908)
    plt.show()

def graph_new_attack_focused(dic,results,res_key,y_lim,mode='test',reverse_row=False,baseline=None,sort_key=None):
    """
    Simplified and robust version: Each attack gets its own subplot, aggregators as lines
    """
    row = 'dataset_dist'
    row_dic = {'iid':'IID','dirichlet':'non-IID','noniid':'non-IID'}
    
    # Process aggregator names
    new_vec = []
    for val in dic['aggr']:
        if val == 'ccs':
            val = 'S-CC'
        elif val == 'krum':
            val = 'M-Krum'
        elif val == 'bulyan':
            val = 'Bulyan'
        elif val == 'fl_trust':
            val = 'FL-Trust'
        elif val == 'krum2':
            val = 'Krum'
        elif val == 'tm_perfect':
            val = 'TM-Oracle'
        elif val == 'tm_history':
            val = 'TM-History'
        elif val == 'tm_capped':
            val = 'CliM'
        elif val == 'tm_cheby':
            val = 'TM-Cheby'
        else:
            val = val.upper()
        new_vec.append(val)
    
    # Process attack names
    new_atk = []
    for num, atk in zip(dic['traitor'], dic['attack']):
        if num == 0:
            atk = 'NoATK'
        new_atk.append(atk)
    
    dic['aggr'] = new_vec
    dic['attack'] = new_atk

    # Simple color mapping for aggregators
    aggr_colors = {
        'FEDAVG': 'blue', 'S-CC': 'red', 'M-Krum': 'orange', 'Krum': 'purple',
        'Bulyan': 'magenta', 'FL-Trust': 'green', 'TM-Oracle': 'red', 'TM-History': 'olive',
        'CliM': 'indigo', 'TM-Cheby': 'darkred', 'TM': 'brown', 'MEDIAN': 'pink'
    }
    
    # Attack name mapping
    atk_map = {'alie': 'ALIE', 'ipm': 'IPM', 'bit_flip': 'Bit flip', 'label_flip': 'Label flip', 
               'cw': 'C&W', 'reloc': 'ROP', 'sparse': 'HSA', 'NoATK': 'Baseline', 
               'minmax': 'MinMax', 'minsum': 'MinSum'}
    
    # Get unique attacks and distributions
    avail_atk = sorted(list(set(dic['attack'])))
    avail_row = sorted(list(set(dic[row])))
    if reverse_row:
        avail_row = list(reversed(avail_row))
    
    # Create data structure: attack+distribution -> list of (aggregator, results)
    data_map = {}
    for r, aggr, attack, num_val in zip(dic[row], dic['aggr'], dic['attack'], results[res_key]):
        key = f'{r}+{attack}'
        if key not in data_map:
            data_map[key] = []
        data_map[key].append((aggr, num_val))
    
    # Create subplot layout
    row_size = 2.5 * len(avail_row)
    figsize = (20, row_size)
    matplotlib.style.use('seaborn-v0_8')
    plt.style.use("seaborn-v0_8")
    fig, axs = plt.subplots(len(avail_row), len(avail_atk), figsize=figsize)
    
    # Handle single subplot case
    if len(avail_row) == 1 and len(avail_atk) == 1:
        axs = [axs]
    elif len(avail_row) == 1 or len(avail_atk) == 1:
        axs = axs.flatten()
    else:
        axs = axs.flatten()
    
    # Track legend entries
    legend_data = {}  # aggregator_name -> line_object
    
    count = 0
    for row_idx, row_val in enumerate(avail_row):
        for atk_idx, atk_val in enumerate(avail_atk):
            ax = axs[count]
            
            # Set up subplot
            y_ticks = np.arange(y_lim[0], y_lim[1] + 1, 20)
            x_ticks = np.arange(0, 101, 20)
            ax.set_yticks(y_ticks)
            ax.set_xticks(x_ticks)
            ax.grid(True, linestyle='-', linewidth=1, alpha=0.6, color='white')
            ax.set_ylim(y_lim)
            ax.set_xlim(0, 100)
            
            # Add baseline if specified
            if baseline is not None:
                ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
            
            # Get data for this attack+distribution combination
            key = f'{row_val}+{atk_val}'
            if key in data_map:
                for aggr_name, acc_data in data_map[key]:
                    # Get color
                    color = aggr_colors.get(aggr_name, 'black')
                    
                    # Plot line
                    x = range(1, len(acc_data) + 1)
                    line_obj = ax.plot(x, acc_data, color=color, linestyle='-', linewidth=1.5, label=aggr_name)
                    
                    # Store for legend (only once per aggregator)
                    if aggr_name not in legend_data:
                        legend_data[aggr_name] = line_obj[0]
            
            # Set subplot title
            attack_title = atk_map.get(atk_val, atk_val)
            if count < len(avail_atk):
                ax.set_title(attack_title, fontweight="bold", size=18)
            
            # Y-axis labels
            if count % len(avail_atk) == 0:
                y_label = f'Test Accuracy \n {row_dic.get(row_val, row_val)}'
                ax.set_ylabel(y_label, fontsize=19.0)
                ax.tick_params(axis='y', labelsize=16)
            else:
                ax.tick_params(axis='y', labelleft=False)
            
            # X-axis labels
            if count >= (len(avail_row) - 1) * len(avail_atk):
                ax.set_xlabel('Epoch', fontsize=19)
                ax.tick_params(axis='x', labelsize=14)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            
            count += 1
    
    # Create legend from collected data
    if legend_data:
        lines = list(legend_data.values())
        labels = list(legend_data.keys())
        
        fig.legend(lines, labels,
                   loc='center right',
                   borderaxespad=0.1,
                   frameon=False,
                   fontsize=14.5,
                   title='AGR',
                   title_fontsize=14.5)
    
    # Create baseline legend if specified
    if baseline is not None:
        from matplotlib.lines import Line2D
        baseline_legend_line = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        baseline_legend = fig.legend([baseline_legend_line], ['Baseline'], 
                                   loc='upper right', 
                                   bbox_to_anchor=(.98, 0.80),
                                   borderaxespad=0.1,
                                   frameon=False,
                                   fontsize=14,
                                   title_fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.908)
    plt.show()


def graph_prune_method(dic,results,res_key,y_lim,mode='test',reverse_row=False):
    row = 'pruning_factor'
    new_vec = []
    r_ = dic[row]
    r = []
    for f in r_:
      if f not in r:
        r.append(f)
    r = sorted(r)
    row_dic = {str(i):i for i in r}
    for val in dic['aggr']:
        if val == 'ccs':
            val = 'S-CC'
        elif val == 'krum':
            val = 'M-Krum'
        elif val == 'bulyan':
            val = 'Bulyan'
        else:
            val = val.upper()
        new_vec.append(val)

    dic['aggr'] = new_vec
    label_dic = {byz:True for byz in dic['prune_method']}

    atk_colors = {'FORCE': 'r', 'Synflow': 'b', 'LAMP': 'orange'
        , 'Random': 'm', 'Uniform+': 'g', 'GRASP': 'k', 'SNIP': 'c','Random+':''}
    atk = {3: 'FORCE', 4: 'Synflow',
           5: 'GRASP', 6: 'SNIP'
        , 7: 'LAMP', 10: 'Uniform+', 11: 'Random',12: 'Random+'}
    avail_aggr = []
    for a in dic['aggr']:
        if a not in avail_aggr:
            avail_aggr.append(a)

    avail_row = []
    for a in dic[row]:
        if a not in avail_row:
            avail_row.append(a)
    if reverse_row:
        avail_row = list(reversed(avail_row)) #### IF needed
    combined = []
    avail_aggr = sorted(avail_aggr)
    for r in avail_row:
        for aggr in avail_aggr:
            combined.append('{}+{}'.format(r,aggr))
    final_dict = {d:[] for d in combined}
    for r,aggr,attack,num_val in zip(dic[row],dic['aggr'],dic['prune_method'],results[res_key]):
        res = [attack,num_val]
        final_dict['{}+{}'.format(r,aggr)].append(res)
    row_size = 3 *len(avail_row)
    figsize = (17, row_size)
    matplotlib.style.use('seaborn')
    fig, axs = plt.subplots(len(avail_row), len(avail_aggr), figsize=figsize)
    axs = axs.flatten()
    lines,legends = [], []
    count = 0
    for ax,sim in zip(axs,final_dict):
        for pert,acc in (final_dict[sim]):
            pert = int(pert)
            try:
                perturbation = atk[pert]
            except:
                perturbation = pert
            color = atk_colors[perturbation]
            #print(perturbation,color)
            linestyle = '--' if perturbation == 'Baseline' else '-'
            x = range(1, len(acc) + 1)
            ax.set_ylim(y_lim)
            ax.set_xlim(0,100)
            #ax.set_facecolor('xkcd:light grey')
            if label_dic[pert]:
                lines.append(ax.plot(x,acc,color=color,linestyle=linestyle,linewidth=1.5))
                legends.append(perturbation)
                label_dic[pert] = False
            else:
                ax.plot(x, acc, color=color,linestyle=linestyle, linewidth=1.5)
        row_str, aggr_str = sim.split('+')
        title = '{}'.format(aggr_str)
        if count< len(avail_aggr):
            ax.set_title(title, fontweight="bold", size=18)  # Title
        if count % len(avail_aggr) >0 and count % len(avail_aggr) != len(avail_aggr):
            ax.set_yticklabels([])
        else:
            #ax.set(ylabel='Test Accuracy')
            if mode =='test':
                y_label = 'Test Accuracy \n {}'.format(row_dic[row_str])
                ax.set_ylabel(y_label, fontsize=19.0)  # Y label

            else:
                ax.set_ylabel('Train loss', fontsize=19.0)  # Y label
            ax.label_outer()
            ax.tick_params(axis='y', labelsize=16)
        if count < (len(avail_row) -1) * len(avail_aggr):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Epoch', fontsize=19)  # X label
            ax.tick_params(axis='x', labelsize=14)
            ax.label_outer()
        count +=1
    fig.legend(lines,  # The line objects
               labels=legends,  # The labels for each line
               loc='center right',  # Position of legend "center right" # 'upper center'
               borderaxespad=0.1,
               frameon=False,
               fontsize=14.5,
               title='ATK',
               title_fontsize=14.5,
               # ncol =3
               )
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(right=0.908)
    plt.show()


def graph_new_atk(dic,results,res_key,y_lim,mode='test',reverse_row=False):
    row = 'num_client'
    new_vec = []
    #row_dic = {'iid':'IID','dirichlet':'non-IID'}
    #row_dic = {'0.0': '\u03B2:{}'.format(0), '0.9': '\u03B2:{}'.format(0.9), '0.99': '\u03B2:{}'.format(0.99)}
    row_dic = {'10.0': 'K={}'.format(10), '25.0': 'K={}'.format(25), '50.0': 'K={}'.format(50),'100.0': 'K={}'.format(100)}
    for val,tau in zip(dic['aggr'],dic['tau']):
        if val == 'ccs':
            val = 'S-CC'
        elif val == 'krum':
            val = 'M-Krum'
        elif val == 'bulyan':
            val = 'Bulyan'
        elif val == 'cc':
                val = 'CC \u03C4={}'.format(tau)
            #val = 'CC L={}'.format(int(n_iter))
        else:
            val = val.upper()
        new_vec.append(val)
    new_atk = []

    for num, atk in zip(dic['traitor'], dic['attack']):
        if num == 0:
            atk = 'NoATK'
        new_atk.append(atk)
    dic['aggr'] = new_vec
    dic['attack'] = new_atk
    label_dic = {byz:True for byz in dic['attack']}

    atk_colors = {'ROP': 'r', 'ALIE': 'b', 'IPM': 'orange'
        , 'Label flip': 'm', 'Bit flip': 'g', 'FedAVG': 'k', 'Baseline': 'k', 'Sparse': 'c','Sparse-R':'k'}
    atk = {'alie': 'ALIE', 'ipm': 'IPM',
           'bit_flip': 'Bit flip', 'label_flip': 'Label flip'
        , 'reloc': 'ROP', 'ROP-S': 'ROP-S', 'sparse': 'Sparse',
           'NoATK': 'Baseline','Sparse-R':'Sparse-R'}
    avail_aggr = []
    for a in dic['aggr']:
        if a not in avail_aggr:
            avail_aggr.append(a)

    avail_row = []
    for a in dic[row]:
        if a not in avail_row:
            avail_row.append(a)
    if reverse_row:
        avail_row = list(reversed(avail_row)) #### IF needed
        #avail_row = [avail_row[1],avail_row[0],avail_row[2]] # hardfixes
        #avail_row = [avail_row[2], avail_row[1], avail_row[2],avail_row[3]]  # hardfixes
        #avail_row = [avail_row[1], avail_row[2], avail_row[3], avail_row[0]]  # hardfixes
        #avail_row = [avail_row[0], avail_row[3], avail_row[1], avail_row[2]]  # hardfixes
        avail_row = [avail_row[0], avail_row[3], avail_row[1], avail_row[2]]  # hardfixes
    combined = []
    avail_aggr = sorted(avail_aggr)
    for r in avail_row:
        for aggr in avail_aggr:
            combined.append('{}+{}'.format(r,aggr))
    final_dict = {d:[] for d in combined}
    for r,aggr,attack,num_val in zip(dic[row],dic['aggr'],dic['attack'],results[res_key]):
        res = [attack,num_val]
        #print(res)
        final_dict['{}+{}'.format(r,aggr)].append(res)
    row_size = 3.2 *len(avail_row)
    figsize = (17, row_size)
    matplotlib.style.use('seaborn')
    fig, axs = plt.subplots(len(avail_row), len(avail_aggr), figsize=figsize)
    axs = axs.flatten()
    lines,legends = [], []
    count = 0
    for ax,sim in zip(axs,final_dict):
        for pert,acc in (final_dict[sim]):
            try:
                perturbation = atk[pert]
            except:
                perturbation = pert
            color = atk_colors[perturbation]
            #print(perturbation,color)
            linestyle = '--' if perturbation == 'Baseline' else '-'
            x = range(1, len(acc) + 1)
            ax.set_ylim(y_lim)
            ax.set_xlim(0,100)
            #ax.set_facecolor('xkcd:light grey')
            if label_dic[pert]:
                lines.append(ax.plot(x,acc,color=color,linestyle=linestyle,linewidth=1.5))
                legends.append(perturbation)
                label_dic[pert] = False
            else:
                ax.plot(x, acc, color=color,linestyle=linestyle, linewidth=1.5)
        row_str, aggr_str = sim.split('+')
        title = '{}'.format(aggr_str)
        if count< len(avail_aggr):
            ax.set_title(title, fontweight="bold", size=18)  # Title
        if count % len(avail_aggr) >0 and count % len(avail_aggr) != len(avail_aggr):
            ax.set_yticklabels([])
        else:
            #ax.set(ylabel='Test Accuracy')
            if mode =='test':
                y_label = 'Test Accuracy \n {}'.format(row_dic[row_str])
                ax.set_ylabel(y_label, fontsize=19.0)  # Y label

            else:
                ax.set_ylabel('Train loss', fontsize=19.0)  # Y label
            ax.label_outer()
            ax.tick_params(axis='y', labelsize=16)
        if count < (len(avail_row) -1) * len(avail_aggr):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Epoch', fontsize=19)  # X label
            ax.tick_params(axis='x', labelsize=14)
            ax.label_outer()
        count +=1
    fig.legend(lines,  # The line objects
               labels=legends,  # The labels for each line
               loc='center right',  # Position of legend "center right" # 'upper center'
               borderaxespad=0.1,
               frameon=False,
               fontsize=14.5,
               title='ATK',
               title_fontsize=14.5,
               # ncol =3
               )
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(right=0.908)
    plt.show()


def alie_alt():
    matplotlib.style.use('seaborn')
    alie_res = np.load('Results/alie_altsign/ATK_alie-Def_tm-dist_cifar10_iid-B_0-Z_None-580/vecs/Test_acc.npy')
    alie_alt = np.load('Results/alie_altsign/ATK_alie-Def_tm-dist_cifar10_iid-B_0-Z_None-5147-ALTERNATINGSIGN/vecs/Test_acc.npy')
    plt.plot(alie_res,color='blue',
     linewidth=2,label='ALIE')
    plt.plot(alie_alt,color='orange', linestyle='dashed',
     linewidth=2,label='ALIE +/-')
    plt.ylabel('Test Accuracy',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.tick_params(labelsize=16)
    plt.legend(loc=4,fontsize=16,)
    plt.tight_layout()
    plt.show()

intervals = 1
#pick = {'Lmomentum':[0,0.9,0.99],'dataset_dist':['iid','dirichlet']}
#pick = {'pi':[1],'dataset_dist':['dirichlet'],'Lmomentum':[.9]}
pick = None
#pick = {'attack':['alie','reloc','ipm'],'dataset_dist':['iid']}
#pick = {'dataset_dist':['iid'],'aggr':['tm'],'Lmomentum':[0]}
#pick = {'num_client':[100]}
#pick = {'sparse_scale':[1.5],'pruning_factor':[0.005],'sparse_cfg':[39]}
#select = ['aggr','Lmomentum','attack','tau','traitor']
#select = ['aggr','Lmomentum','attack','tau','traitor','dataset_dist','z_max']
#select = ['dataset_dist','Lmomentum','attack','buck_len','multi_clip','buck_avg']
#select = ['dataset_dist','Lmomentum','aggr','attack','pi','lamb','angle']
#select = ['aggr','Lmomentum','tau','dataset_name']
#select = ['aggr','Lmomentum','tau','traitor','attack','lamb']
#select = ['aggr','Lmomentum','tau','traitor','attack','z_max']
#select = ['dataset_dist','aggr','Lmomentum','pruning_factor','sparse_scale','pi','lamb','angle'] #sparse
#select = ['dataset_dist','aggr','Lmomentum','sparse_cfg','pruning_factor','z_max','sparse_scale','lamb','init'] #sparse
#select = ['dataset_dist','aggr','Lmomentum','sparse_cfg','pruning_factor','sparse_scale','init','num_proj','num_steps','z_max'] #sparse
#select = ['dataset_dist','aggr','Lmomentum','sparse_cfg','pruning_factor','sparse_scale','init','num_proj','num_steps','z_max','nn_name'] #sparse
#select = ['dataset_dist','aggr','attack','Lmomentum','sparse_cfg','pruning_factor','sparse_scale','z_max','lamb'] #sparse
#select = ['dataset_dist','aggr','attack','sparse_cfg','weight_init','prune_method','random_mask','pruning_factor','sparse_scale','z_max'] #sparse
#select = ['dataset_dist','aggr','attack','sparse_cfg','weight_init','prune_method','pruning_factor','sparse_scale','n_iter'] #random
#select = ['dataset_dist','aggr','attack','sparse_cfg','weight_init','prune_method','pruning_factor','sparse_scale','num_batches','prune_bs','gas_p','num_steps'] #random
#select = ['dataset_dist','aggr','attack','sparse_cfg','weight_init','prune_method','pruning_factor','sparse_scale','num_batches','prune_bs','gas_p','inout_layers']
#select = ['dataset_dist','aggr','attack','sparse_cfg','prune_method','pruning_factor','sparse_scale','num_batches','prune_bs','gas_p','prune_dataset_split','mask_scope']
select = ['dataset_dist','aggr','attack','sparse_cfg','prune_method','pruning_factor','num_steps','num_batches','prune_bs','gas_p','fc_treshold','load_mask']
#select = ['dataset_dist','aggr','attack','alie_z_max']
#select = ['dataset_dist','Lmomentum','aggr','tau','attack','traitor','epsilon','num_client'] #graph3
#select = ['dataset_dist','aggr','attack','traitor','random_mask'] #forRandom-msan
#select = ['dataset_dist','aggr','attack','z_max','angle','pi'] #angle-study
#select = ['dataset_dist','aggr','attack','angle','gas_p'] #Bench
select = ['dataset_dist','Lmomentum','aggr','tau','attack','traitor'] #graph3
select = ['dataset_dist','buck_len','buck_len_ecc','bucket_op','ref_fixed'] #graph3
select = ['dataset_dist','aggr','attack','cl_part','traitor','MITM'] #SP
select = ['dataset_dist','aggr','attack','prune_method','pruning_factor','fc_treshold','conv_threshold','num_steps','sparse_scale','bn_scale','weight_init','min_threshold','keep_orig_weights'] #new_prune
select = ['dataset_dist','aggr','attack','num_clustering','bucket_shift','shift_amount'] #SP
select = ['dataset_dist','aggr','attack','buck_len','buck_avg'] #SP
select = ['dataset_dist','aggr','attack','load_mask','traitor'] #SP
select = ['dataset_dist','aggr','attack','prune_method','pruning_factor','min_threshold','weight_init'] #SP
select = ['dataset_dist','aggr','attack','hybrid_aggregator_list','prune_method','pruning_factor','min_threshold','sparse_th','weight_init','init'] #SP

#select = ['aggr','num_client','avg_aggr_time','traitor'] #graph3
#select = ['dataset_dist','aggr','attack','sparse_cfg','traitor','prune_method','pruning_factor','sparse_cfg']
#file_loc = 'Results/Alie-Medians'
file_loc = 'Results/foundation'
#file_loc = 'Results/sparse-cfg23'
get = 'Train_losses,Test_acc,Clipping'.split(',')[1]
f_name = file_loc.split('/')[-1] + '-{}'.format(get)
format = 'png'
save_loc = path.join('/home/kerem/Desktop/BYZ_figs',f_name) +'.{}'.format(format)
dic, results = load_results(file_loc, select,pick=pick,latex=True)
#save_fig = {'fname':save_loc,'format':format,'dpi':400}
#graph(get,results,labels,save_fig=None)
#graph_2(dic,results,get,y_lim=[0,100],mode='test')
#graph_z(dic,results,get,y_lim=[0,100],mode='test')
#graph_3(dic,results,get,y_lim=[0,100],mode='test')
#graph_new(dic,results,get,y_lim=[10,100],mode='test',reverse_row=False,baseline=88)
#graph_new_attack_focused(dic,results,get,y_lim=[10,100],mode='test',reverse_row=True,baseline=88)

# Baseline appears in separate 'Reference' legend at top right
# Attack types appear in 'ATK' legend at center right
#graph_custom(dic,results,get,y_lim=[0,100],mode='test',reverse_row=False)
#graph_prune_method(dic,results,get,y_lim=[0,100],mode='test',reverse_row=False)
#graph_random_prune(dic,results,get,y_lim=[0,100],mode='test',reverse_row=False)
#graph_new_atk(dic,results,get,y_lim=[0,100],mode='test',reverse_row=True)

