import pickle
def plot_tests(xlabel='Epochs',ylabel='Cost'):
    trial_parameters={'Gauss':(50,1.0,'gaussian',0,0),
                      'Berno':(50,1.0,'bernoulli',0,0),
                      'Berno+BN':(50,1.0,'bernoulli',1,0),
                      'Berno+BN+WU':(50,1.0,'bernoulli',1,1)}
    x_max=y_max=0
    trials={}
    for trial_name in trial_parameters:
        data=test_points(*trial_parameters[trial_name])
        data_x_max=data['cost'][1]
        if data_x_max>x_max:
            x_max=data_x_max
        data_y_max=data['cost'][2]
        if data_y_max>y_max:
            y_max=data_y_max
        trials[trial_name]=data
    color_list=['red','orange','yellow','green','blue','purple']
    print '\\documentclass{article}'
    print '\\usepackage{pgfplots}'
    print '\\begin{document}'
    print '\\begin{tikzpicture}\\begin{axis}'
    print '    [title={Cost for parameter settings},'
    print '    xlabel={%s},ylabel={%s},' % (xlabel,ylabel)
    print '    xmin=0, xmax={},ymin=0, ymax={},]'.format(x_max*1.1,y_max*1.1)
    #print '    xtick={0,20,40,60,80,100},ytick={0,20,40,60,80,100,120},'
    #print '    legend pos=north west,ymajorgrids=true,grid style=dashed,]'
    color_index=0
    legend_string=''
    for trial_name in trials:
        print '    \\addplot[color={},mark=square,]'.format(color_list[color_index])
        color_index+=1
        print '    coordinates {'
        print trials[trial_name]['cost'][0]
        print '    };'
        legend_string+=trial_name+','
    print '    \\legend{%s}' % legend_string
    print '\\end{axis}\\end{tikzpicture}'
    print '\\end{document}'

def print_points(data,dtype=None):
    ret=''
    for i in xrange(len(data)):
        if dtype:
            ret+="({},{})".format(i,dtype(data[i]))
    return (ret,len(data),float(max(data)))

def test_points(z_dim,keep_prob,gen_dist,b_normal,warmup,t_epochs=20):
    namestring='../trials_all/trial.{}.{}.{}.{}.{}.pkl'.format(z_dim,keep_prob,gen_dist,b_normal,warmup,t_epochs)
    #{'cost':cost_list,'covar':covar_list}
    data=pickle.load(open(namestring,'r'))
    return {'cost':print_points(data['cost'],float),'covar':0}

plot_tests()
# In[ ]:
#Initial tests to demonstrate features
#plot_test(10,1.0,'gaussian',0,0)
#run_test(10,1.0,'bernoulli',0,0)
#run_test(10,1.0,'bernoulli',1,0)
#run_test(10,1.0,'bernoulli',1,1)
#Test dropout rate
#for keep in [1.0,0.9,0.8,0.7,0.6]:
#    run_test(10,keep,'bernoulli',1,1)
#Test N_z
#for n_z in [2,5,10,20]:
#    run_test(n_z,1.0,'bernoulli',1,1)
#Test warmup over dimensions
#for dim in [5,20]:
#    run_test(dim,1.0,'bernoulli',1,0)
#    run_test(dim,1.0,'bernoulli',1,1)
#Test dropout over dimensions
#for dim in [5,20]:
#    run_test(dim,1.0,'bernoulli',0,1)
#    run_test(dim,1.0,'bernoulli',1,1)

