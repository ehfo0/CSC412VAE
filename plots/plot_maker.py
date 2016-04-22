import pickle
import numpy as np
import sys
def plot_tests(trial_parameters,xlabel='Epochs',ylabel='Cost',title='Cost for parameter settings',axis='axis',plot='cost'):
    x_max=y_max=0
    y_min=sys.maxint
    x_min=0
    trials={}
    for trial_name in trial_parameters:
        data=test_points(*trial_parameters[trial_name])
        data_x=data[plot][1]
        if data_x>x_max:
            x_max=data_x
        data_y=data[plot][2]
        if data_y>y_max:
            y_max=data_y
        data_y=data[plot][3]
        if data_y<y_min:
            y_min=data_y
        trials[trial_name]=data
    color_list=['red','green','blue','orange','purple']
    print '\\begin{tikzpicture}\\begin{%s}' % axis
    print '    [title={%s},' % title
    print '    xlabel={%s},ylabel={%s},' % (xlabel,ylabel)
    print '    xmin={}, xmax={},ymin={}, ymax={},]'.format(x_min*0.9,x_max*1.1,y_min*0.9,y_max*1.1)
    #print '    xtick={0,20,40,60,80,100},ytick={0,20,40,60,80,100,120},'
    #print '    legend pos=north west,ymajorgrids=true,grid style=dashed,]'
    color_index=0
    legend_string=''
    for trial_name in sorted(trials.keys()):
        print '    \\addplot[color={},mark=square,]'.format(color_list[color_index])
        color_index+=1
        print '    coordinates {'
        print trials[trial_name][plot][0]
        print '    };'
        legend_string+=trial_name+','
    print '    \\legend{%s}' % legend_string
    print '\\end{%s}\\end{tikzpicture}\\\\' % axis

def print_points(data,dtype=None):
    ret=''
    for i in xrange(len(data)):
        if dtype:
            ret+="({},{})".format(i,dtype(data[i]))
        else:
            ret+="({},{})".format(i,data[i])
    return (ret,len(data),float(max(data)),float(min(data)))

def test_points(z_dim,keep_prob,gen_dist,b_normal,warmup,t_epochs=20):
    namestring='../trials_all/trial.{}.{}.{}.{}.{}.pkl'.format(z_dim,keep_prob,gen_dist,b_normal,warmup,t_epochs)
    #{'cost':cost_list,'covar':covar_list}
    data=pickle.load(open(namestring,'r'))
    threshold=1e-2
    return {'cost':print_points(data['cost'],float),'covar':print_points(np.greater(data['covar'],threshold).sum(1))}

#HEADER
print '\\documentclass{article}'
print '\\usepackage{pgfplots}'
print '\\begin{document}'
#TESTS
plot_tests({'Gauss':(50,1.0,'gaussian',0,0),
            'Berno':(50,1.0,'bernoulli',0,0)})
plot_tests({'Berno':(50,1.0,'bernoulli',0,0),
            'Berno+BN':(50,1.0,'bernoulli',1,0),
            'Berno+BN+WU':(50,1.0,'bernoulli',1,1)},
           axis='semilogyaxis')
plot_tests({'Gauss':(50,1.0,'gaussian',0,0),
            'Berno':(50,1.0,'bernoulli',0,0)},
           plot='covar',ylabel='Active Dimensions')
plot_tests({'Berno':(50,1.0,'bernoulli',0,0),
            'Berno+BN':(50,1.0,'bernoulli',1,0),
            'Berno+BN+WU':(50,1.0,'bernoulli',1,1)},
           plot='covar',ylabel='Active Dimensions',title='Active Dimensions for Parameters')
#Test dropout rate
params={}
for keep in [1.0,0.9,0.8,0.7,0.6]:
    params['DO='+str(keep)]=(50,keep,'bernoulli',1,1)
plot_tests(params,plot='cost')
plot_tests(params,plot='covar',ylabel='Active Dimensions',title='Active Dimensions for Parameters')
#FOOTER
print '\\end{document}'
# In[ ]:
#Initial tests to demonstrate features
#plot_test(10,1.0,'gaussian',0,0)
#run_test(10,1.0,'bernoulli',0,0)
#run_test(10,1.0,'bernoulli',1,0)
#run_test(10,1.0,'bernoulli',1,1)

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

