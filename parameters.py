import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # technical params
    parser.add_argument('--trials', type=int, default=1, help='number of trials')
    parser.add_argument('--cuda', type=bool, default=True, help='Use cuda as device')
    parser.add_argument('--worker_per_device', type=int, default=1, help='parallel processes per device')
    parser.add_argument('--excluded_gpus', type=list, default=[], help='bypassed gpus')
    parser.add_argument('--kuacc', type=bool, default=False, help='Koç kuis cluster')

    # Federated params
    parser.add_argument('--global_epoch', type=int, default=10, help='total cumulative epoch')
    parser.add_argument('--localIter', type=int, default=1, help='Local Epoch')
    parser.add_argument('--num_client', type=int, default=25, help='number of clients')
    parser.add_argument('--cl_part', type=float, default=1., help='participation ratio of the clients')
    parser.add_argument('--traitor', type=float, default=0.2, help='traitor ratio')
    parser.add_argument('--attack', type=str, default='alie', help='see Attacks')
    parser.add_argument('--aggr', type=str, default='bulyan', help='see aggregators.py')
    parser.add_argument('--hybrid_aggregator_list', type=str, default='cc+tm', 
                        help='List of aggregators for hybrid aggregation last one does final aggregation')
    parser.add_argument('--embd_momentum', type=bool, default=False, help='FedADC embedded momentum')
    parser.add_argument('--early_stop', type=bool, default=False, help='Early stop function')
    parser.add_argument('--MITM', type=bool, default=True, help='Adversary capable of man-in-middle-attack')
    parser.add_argument('--bucketing', type=bool, default=False, help='Bucket the clients before aggregation')
    parser.add_argument('--bucket_type', type=str, default='Random', 
                        help='Type of bucketing, Random, Cosine Distance, L2 distance')
    parser.add_argument('--bucket_size', type=int, default=3, help='bucket length for sequential cc')


    # Defence params
    parser.add_argument('--tau', type=list, default=[1], help='Radius of the ball for CC aggregator')
    parser.add_argument('--n_iter', type=list, default=[1], help='number of iteration for cc aggr')
    parser.add_argument('--buck_rand', type=bool, default=False, help='bucket random selection for sequential cc')
    parser.add_argument('--buck_len', type=int, default=3, help='bucket length for sequential cc')
    parser.add_argument('--buck_len_ecc', type=int, default=3, help='bucket length for sequential cc in ecc')
    parser.add_argument('--buck_avg', type=bool, default=False, help='average the bucket for sequential cc')
    parser.add_argument('--multi_clip', type=bool, default=False, help='average the bucket for sequential cc')
    parser.add_argument('--bucket_op', type=str, default=None, help='Operations if last bucket is size 1 [None, merge, split]')
    parser.add_argument('--ref_fixed', type=bool, default=False, help='static reference for sequential ECC cc')
    parser.add_argument('--shuffle_bucket_order', type=bool, default=False, help='shuffle the bucket order for sequential cc L2')
    parser.add_argument('--combine_bucket', type=bool, default=False, help='combine the bucket for sequential cc L2')
    parser.add_argument('--T', type=int, default=3, help='RFA inner iteration')
    parser.add_argument('--nu', type=float, default=0.1, help='RFA norm budget')
    parser.add_argument('--gas_p', type=int, default=1000, help='Number of chunks for GAS')
    parser.add_argument('--fg_use_memory', type=bool, default=True, help='FoolsGold: Whether to use gradient history for detection')
    parser.add_argument('--fg_memory_size', type=int, default=10, help='FoolsGold: Number of previous rounds to remember')
    parser.add_argument('--fg_epsilon', type=float, default=1e-5, help='FoolsGold: Small constant for numerical stability')
    parser.add_argument('--cheby_k_sigma', type=float, default=1.0, help='Chebyshev TM: Number of standard deviation scale for outlier detection')
    parser.add_argument('--foundation_num_synthetic', type=int, default=2, help='Number of synthetic updates to generate for FoundationFL')
    parser.add_argument('--lalambda_n', type=float, default=1.0, help='LASA: Magnitude Detection threshold for Byzantine clients')
    parser.add_argument('--lalambda_s', type=float, default=1.0, help='LASA: Sign Detection threshold for Byzantine clients')
    parser.add_argument('--lasa_sparsity_ratio', type=float, default=0.3, help='LASA: Sparsity ratio for client updates')

    # Experimental defence params (Might remove in the future)
    parser.add_argument('--num_clustering', type=int, default=3, help='number of clustering for cluster based aggregators')
    parser.add_argument('--bucket_shift', type=str, default='sequential', help='bucket shift type for cluster based aggregators, sequential or random')
    parser.add_argument('--shift_amount', type=int, default=1, help='bucket shift amount for sequential bucket shift')  
    parser.add_argument('--buck_len_l2', type=int, default=3, help='bucket shift amount for sequential bucket shift')
    parser.add_argument('--apply_TM', type=bool, default=False, help='')
    parser.add_argument('--seq_update', type=bool, default=False, help='')


    # attack params
    parser.add_argument('--z_max', type=list, default=[None], help='attack scale,none for auto generate')
    parser.add_argument('--alie_z_max', type=float, default=None, help='attack scale for ALIE attack')
    parser.add_argument('--nestrov_attack', type=bool, default=False, help='clean step first- For non-omniscient attacks')
    parser.add_argument('--epsilon', type=float, default=0.2, help='IPM attack scale')
    parser.add_argument('--pert_vec', type=str, default='std', help='[unit_vec,sign,std] for minmax and minsum attacks')
    parser.add_argument('--delta_coeff', type=list, default=[.9], help='[unit_vec,sign,std] for minmax and minsum attacks')


    # modular attack
    parser.add_argument('--pi', type=list, default=[0], help='location of the attack,1 for full relocation to aggr reference')
    parser.add_argument('--angle', type=list, default=[270], help='angle of the pert, 180,90 and none')
    parser.add_argument('--lamb', type=list, default=[0.9], help='refence point for attack if angle is not none')


    # optimiser related
    parser.add_argument('--opt', type=str, default='sgd', help='name of the optimiser')
    parser.add_argument('--lr', type=float, default=0.1, help='learning_rate')
    parser.add_argument('--lr_decay', type=list, default=[75], help='lr drop at given epoch')
    parser.add_argument('--wd', type=float, default=0, help='weight decay Value')
    parser.add_argument('--Lmomentum', type=float, default=0.9, help='Local Momentum for SGD')
    parser.add_argument('--betas', type=tuple, default=(0.9,0.999), help='betas for adam and adamw opts')
    parser.add_argument('--worker_momentum', type=bool, default=True, help='adam like gradiant multiplier for SGD (1-Lmomentum)')
    parser.add_argument('--nesterov', type=bool, default=False, help='nestrov momentum for Local SGD steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='clip gradient norm; set 0 or negative to disable')

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='mnist', help='see data_loader.py')
    parser.add_argument('--dataset_dist', type=list, default=['iid'],
                        help='distribution of dataset; iid or sort_part, dirichlet')
    parser.add_argument('--numb_cls_usr', type=int, default=2,
                        help='number of label type per client if sort_part selected')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='alpha constant for dirichlet dataset_dist,lower for more skewness')
    parser.add_argument('--bs', type=int, default=32, help='batchsize')

    # nn related
    parser.add_argument('--nn_name', type=str, default='mnistnet', help='simplecnn,simplecifar,VGGs resnet(8-9-18-20)')
    parser.add_argument('--weight_init', type=list, default=['-'],
                        help='nn weight init, kn (Kaiming normal) or - (None)')
    parser.add_argument('--norm_type', type=str, default='bn',
                        help='gn (GroupNorm), bn (BatchNorm), - (None)')
    parser.add_argument('--num_groups', type=int, default=32,
                        help='number of groups if GroupNorm selected as norm_type, 1 for LayerNorm')

    # sparse related
    parser.add_argument("--load_mask", type=str, default=None,
                        help='load mask for sparse attack, mask is indices of the non-pruned parameters')
    parser.add_argument("--prune_dataset_split", type=float, default=1.,
                        help='split prune dataset')
    parser.add_argument("--omniscient_pruning", type=bool, default=True,
                        help='Use benign client data to pruning')
    parser.add_argument("--sparse_cfg", type=list, default=[50],
                        help='config of the sparse attack')
    parser.add_argument("--pruning_factor", type=list, default=[0.005], dest="pruning_factor",
                        help='Fraction of connections after pruning')

    parser.add_argument("--sparse_scale", type=list, default=[1.5],
                        help='attack scale for the remaining parameters')
    parser.add_argument("--sparse_sign_rand", type=bool, default=False,
                        help='randomize the sign of the sparse attack')
    parser.add_argument("--sparse_th", type=list, default=['iqr'],
                        help='thresholding method for sparse attack, iqr or z_score or gradient (Usefull on non-IID for exploding variance)')
    parser.add_argument("--prune_method", type=list, default=['force_var_weight'], dest="prune_method",
                        help="""Which pruning method to use:
                                       1->Iter SNIP
                                       2->GRASP-It
                                       3->FORCE 
                                       4-> SynFlow
                                       5-> Grasp
                                       6-> Snip
                                       7-> LAMP
                                       8 -> Erk
                                       9 -> Uniform
                                       10 -> Uniform+
                                       11 -> Random
                                       12 -> Random+
                                       13 -> Random-Layerwise+
                                       14 -> Force-STD  
                                       . """)
    parser.add_argument("--prune_bias", type=bool, default=False,
                        help='prune bias parameters')
    parser.add_argument("--prune_bn", type=bool, default=False,
                        help='prune Batchnorm parameters')
    parser.add_argument("--keep_orig_weights", type=bool, default=True,
                        help='testing parameter, keep the original weights')
    parser.add_argument("--conv_threshold", type=list, default=[.2],
                        help='limit kept params in fc layer in force')
    parser.add_argument("--fc_treshold", type=list, default=[.4],
                        help='limit kept params in fc layer in force')
    parser.add_argument("--min_threshold", type=float, default=.25,
                        help='limit kept params in fc layer in force')
    parser.add_argument("--inout_layers", type=bool, default=False,
                        help='inluce input and output layers in the mask')
    parser.add_argument("--num_steps", type=list, default=[20],
                        help='Number of steps to use with iterative pruning')

    parser.add_argument("--prune_bs", type=list, default=[32],
                        help='Batch_size for pruning')

    parser.add_argument("--mode", type=list, default=['exp'],
                        help='Mode of creating the iterative pruning steps one of "linear" or "exp".')

    parser.add_argument("--num_batches", type=list, default=[1],
                        help='''Number of batches to be used when computing the gradient.
                               If set to -1 they will be averaged over the whole dataset.''')
    parser.add_argument("--force_w", type=float, default=1.,
                        help='saliency score scaling factor for weights')
    parser.add_argument("--force_g", type=float, default=1.,
                        help='saliency score scaling factor for grads')
    parser.add_argument("--force_v", type=float, default=1.,
                        help='saliency score scaling factor for variance')
    parser.add_argument("--init", type=str, default='kn',
                        help='''Which initialization method to use before prunning,normal_kaiming and None
                                Ps model and pruning should have same network init''')

    parser.add_argument("--mask_scope", type=list, default=['local'],
                        help='global,local')

    args = parser.parse_args()
    return args
