import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from fnmatch import fnmatch

#This file prints a latex file to STDOUT and puts shade plots in a folder specified by img_folder
#Redirecting the output to a Tex file in a subfolder is best

#General settings
#dfolder="../trials_post_yujia/"
#dfolder="../trials_piecemeal/"
dfolder="../trials_final/"
img_folder="img/"
latex_path_to_img="../plots/"+img_folder
img_ext='png'
max_epochs=300
global_epoch_cap=299
include_titles=False
#Generates a filename to be read from based on the parameters
def namestring(z_dim,keep_prob,b_normal,warmup,i_weighting,trial_num=None,data_folder=dfolder,ext='.pkl',search=True,epoch_cap=global_epoch_cap,force_distro=None):
    if trial_num!=None:
        #return data_folder+'trial_num.{}.{}.{}.{}.{}.{}{}'.format(trial_num,z_dim,keep_prob,gen_dist,b_normal,warmup,ext)
        #return data_folder+'trial_num.{}.{}.{}.{}.{}{}'.format(trial_num,z_dim,keep_prob,b_normal,warmup,ext)
        query='{}.{}.{}.{}.{}.trial{}*{}{}'.format(z_dim,keep_prob,b_normal,warmup,i_weighting,trial_num,epoch_cap,ext)
        fallback=data_folder+'{}.{}.{}.{}.{}.trial{}{}'.format(z_dim,keep_prob,b_normal,warmup,i_weighting,trial_num,ext)
        if force_distro:
            filename=data_folder+'{}.{}.{}.{}.{}.trial{}.endedat{}.{}{}'.format(z_dim,keep_prob,
                                                                               b_normal,warmup,i_weighting,
                                                                               trial_num,epoch_cap,force_distro,ext)
            errprint("Forced to: "+filename)
            return filename
        if not search:
            return fallback
        for file in os.listdir(data_folder):
            if fnmatch(file,query):
                #errprint('Found: '+data_folder+file)
                return data_folder+file
        errprint("Falling back on filename search: "+query)
        return fallback
    else:
        return data_folder+'trial.{}.{}.{}.{}.{}{}'.format(z_dim,keep_prob,b_normal,warmup,i_weighting,ext)

#Function for making line plots of data. Feeds the parameters to test_points,
# which reads from the data file and parses everything into coordinate pairs
def plot_tests(trial_parameters,xlabel='Epochs',ylabel='$\mathcal{L}(x)$',title='Cost for parameter settings',
               axis='axis',plot='cost',trial_range=[1],bvsg=False):
    #Values for framing the plot
    x_max=y_max=0
    y_min=sys.maxint
    x_min=0
    #Dictionary of processed data
    trials={}
    force_distro=None
    if bvsg:
        force_distro=1
    #Iterate over the dictionary, keyed by name
    for trial_name in trial_parameters:
        #Feed the parameters into the parser to get a clean list of coordinate pairs
        if force_distro:
            if force_distro==1:
                data=test_points(*trial_parameters[trial_name],trial_range=trial_range,force_distro='bernoulli')
                force_distro+=1
            else:
                data=test_points(*trial_parameters[trial_name],trial_range=trial_range,force_distro='gaussian')
        else:
            data=test_points(*trial_parameters[trial_name],trial_range=trial_range)
        #Element 0 is data, elements 1, 2, and 3 are xmax, ymax, and ymin
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
        
    #List of colors for lines, in the order they'll be drawn
    color_list=['red','green','blue','orange','purple','crimson','magenta']
    color_index=0
    #List of trial names that will eventually be put into the legend
    legend_string=''
    #Regenerate a namestring without the pkl extension or the folder name so
    # we have a consistent naming scheme across trials and plot images
    img_ns=title
    outfile_name=img_folder+'%s.%s' % (img_ns,'tex')
    outfile=open(outfile_name,'w')
    #Latex headers for the plot
    outfile.write('\\begin{tikzpicture}\\begin{%s}' % axis)
    comment_string=''
    if not include_titles:
        comment_string='%'
    outfile.write('    [%s title={%s},\n' % (comment_string,title))
    outfile.write('    xlabel={%s},ylabel={%s},' % (xlabel,ylabel))
    outfile.write('    xmin={}, xmax={},ymin={}, ymax={},]'.format(x_min*0.9,x_max*1.1,y_min*0.9,y_max*1.1))
    #outfile.write('    xtick={0,20,40,60,80,100},ytick={0,20,40,60,80,100,120},')
    #outfile.write('    legend pos=north west,ymajorgrids=true,grid style=dashed,]')
    #Sort by name so things print in a nice order
    for trial_name in sorted(trials.keys()):
        #No data, no plot
        if len(trials[trial_name][plot]) and len(trials[trial_name][plot][0]):
            errprint("Publishing plot for "+trial_name)
            #Header for the line to specify color (and other parameters, optionally)
            outfile.write('    \\addplot[color={},]'.format(color_list[color_index]))
            #Move on to the next color in the list
            color_index+=1
            #Print out the list of coordinates for the line
            outfile.write('    coordinates {')
            outfile.write(trials[trial_name][plot][0])
            outfile.write('    };')
            #Append this trial to the legen
            legend_string+=trial_name+','
    #Latex footers for the plot
    outfile.write('    \\legend{%s}' % legend_string)
    outfile.write('\\end{%s}\\end{tikzpicture}\\\\' % axis)
    outfile.flush()
    outfile.close()
    print "\\input{%s}" % ('"%s.%s"' % (latex_path_to_img+img_ns,'tex'))

#Function for making shade plots of data. Feeds the parameters to test_points,
# which reads from the data file and parses everything into coordinate pairs
# trial_range is a list of indices to read from when searching for the average
# (in the case of cost) and longest trials (in the case of covariance)
# Using a range instead of just n_trials means we can skip trials with funny business in them
def shade_tests(trial_parameters,xlabel='Epochs',ylabel='Dimension',title='Activity for parameter settings',trial_range=[1],log_scale=True):
    trials={}
    #Pass the parameters to test_points to parse all the data cleanly for us
    for trial_name in trial_parameters:
        data=test_points(*trial_parameters[trial_name],trial_range=trial_range)
        trials[trial_name]=data
    #Go through everything in sorted order. Not really important for these, but convenient
    for trial_name in sorted(trials.keys()):
        #Transpose is necessary just for orientation. Sort bunches together all of the shaded layers
        grid=np.sort(trials[trial_name]['covar'].T,0)
        if log_scale:
            grid=np.log(grid)
        #vmin and vmax force the scaling parameters for consistency. These have no basis
        im = plt.imshow(grid,cmap="Greys",origin="lower",vmin=-8,vmax=0)
        plt.axis([0,max_epochs,trial_parameters[trial_name][0],0])
        #Colorbar indexes the shades to activity values. The default vertical looks best
        #cb=plt.colorbar(im, orientation='horizontal')
        #cb=plt.colorbar(im)
        #Nice labels for the colorbar, axes, and plot
        #cb.ax.get_yaxis().labelpad = 25
        #cb.ax.set_ylabel('Cov',rotation=90)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if include_titles:
            plt.suptitle("%s %s" % (title,trial_name))
        #Uncomment this to have a matplotlib window pop up with the plot in it
        #plt.show()
        #Regenerate a namestring without the pkl extension or the folder name so
        # we have a consistent naming scheme across trials and plot images
        img_ns=namestring(*trial_parameters[trial_name],data_folder='',ext='',search=False)
        #Save the image to the img folder
        plt.savefig(img_folder+'%s.%s.%s' % (img_ns,trial_name,img_ext), bbox_inches='tight')
        #Clear the way for the next plot
        plt.clf()
        errprint("Publishing shade for "+trial_name)
        #Print out the necessary latex to include this image in the document
        print "\includegraphics[scale=0.75]{{%s.%s}.%s}\\\\" % ((latex_path_to_img+img_ns),trial_name,img_ext)

#Simple function to print coordinate pairs into a string and record their range
# This is what plot_tests is reading for each trial
def print_points(data,dtype=None):
    ret=''
    for i in xrange(len(data)):
        y=data[i]
        if np.isnan(y):
            continue
        if dtype:
            y=dtype(y)
        ret+="({},{})".format(i,y)
    return (ret,len(data),float(max(data)),float(min(data)))

#Parser for the data files that handles NaNs, averaging, and searching for long trials
# Arguments are the test parameters as well as trial_range, which is a list of the
# trial_num's to be read from. This is better than using n_trials because some of them
# have non-NaN related funny business (like the cost suddenly hitting 0) that
# mess with the averaging. In this case, discrimination is a good thing
def test_points(z_dim,keep_prob,b_normal,warmup,i_weighting,trial_range=[1],force_distro=None):
    #We do need the number of trials for building the arrays
    n_trials=len(trial_range)
    #Initialize some empty ndarrays. Covar will be replaced
    full_data={'cost':np.empty([n_trials,max_epochs]),'covar':np.empty([max_epochs,z_dim]),'covar_cut':np.empty([n_trials,max_epochs])}
    #Initialize the maximum usable length of the cost data. This will get shorter
    # due to NaN errors in the trials
    max_len_cost=max_epochs
    #Initialize the current longest known list of covariances. This will get longer
    longest_covar=0
    #Cutoff point for what counts as activity
    threshold=1e-1
    #Confusing notation ahead. i indexes the cost array above, not the trial_num in the file name
    # which is trial_range[i]. This is how we can jump around and skip the funny business
    for i in xrange(0,n_trials):
        #Generate the file name based on the parameters
        nstring=namestring(z_dim,keep_prob,b_normal,warmup,i_weighting,trial_num=trial_range[i],force_distro=force_distro)
        #Read the data from the file. Set up as:
        #{'cost':cost_list,'covar':covar_list}
        data=pickle.load(open(nstring,'r'))

        #PROCESS THE COST ARRAY

        #Read the full cost array (NaN's and all) into an ndarray for processing
        len_cost=len(data['cost'])
        data_cost=np.ndarray(shape=(len_cost),buffer=data['cost'])
        #Store it at its index in the full_data array. It will be averaged with other trials later
        # (this is why we need i)
        full_data['cost'][i][0:len_cost]=data_cost
        #Find out the indices of any NaN's in the array
        nans=np.argwhere(np.isnan(data_cost))
        #If there are any, the first one is where we cut off the array
        if len(nans):
            len_cost=int(min(nans))
        #Find out the indices of any values that are too small, indicating a numerical error
        nans=np.argwhere(data_cost<1e-300)
        #If there are any, the first one is where we cut off the array
        if len(nans):
            len_cost=int(min(nans))
        #If the cutoff for this array is smaller than any we've seen, reduce the
        # useful length of the cost array. This is so we're never averaging with NaN's
        if len_cost<max_len_cost:
            max_len_cost=len_cost

        #PROCESS THE COVARIANCE ARRAY

        #Get the number of epochs for the covariance (handily the first dimension)
        len_covar=len(data['covar'])
        #data['covar'].sum(1) gets the sum of the covariances over dimensions for each epoch.
        # This is only useful because then it becomes a one dimensional array of epochs, and any
        # epoch that generated a NaN in any dimension, now has a NaN in our array
        # isnan turns all these NaN's into True, and argwhere gets the indices of the culprits
        nans=np.argwhere(np.isnan(data['covar'].sum(1)))
        #If there were any naughty NaN's, they're in the nans array
        if len(nans):
            #Use the first instance of NaN as the cutoff
            len_covar=int(min(nans))
        #If, after cutting for NaN's, this is the longest trial we've yet seen, make it the one we keep
        if len_covar>longest_covar:
            data_covar=np.ndarray(shape=(len_covar,z_dim),buffer=data['covar'])
            full_data['covar']=data_covar
            longest_covar=len_covar
        full_data['covar_cut'][i][0:len_covar]=np.greater(full_data['covar'],threshold).sum(1)
    #Average over all of the cost arrays we parsed out, cutting them off and the maximum useful length
    # that we also parsed out (i.e., kill the NaN's)
    avg_cost=np.ndarray(shape=(max_len_cost),buffer=np.mean(full_data['cost'],axis=0))
    #Print points converts the array into a string of coordinate pairs for plotting, plus gets the extreme values
    return {'cost':print_points(avg_cost,float),
            
            #Leave the covariance unadulterated for shade plots
            'covar':full_data['covar'],
            
            #np.greater(full_data['covar'],threshold) picks out only the activity levels above the threshold
            # The sum(1) counts how many actually met the threshold, which is the number we're interested in
            'covar_cut':print_points(np.mean(full_data['covar_cut'],axis=0))}

#Convenience function for when STDOUT is redirected (which should be always)
#Not entirely kosher, but doesn't cause any problems
def errprint(string):
    sys.stderr.write(string+"\n")

if __name__=='__main__':
    #Hack just to make sure the Latex file is set up consistent with the
    # file paths specified at the top of the script
    #Using a redirect is better. E.g., python plot_maker.py > latex/dum.tex
    stdout_hold=sys.stdout
    sys.stdout=open('/home/matt/412/latex/plots.tex','w')
    
    #LATEX HEADER
    print '\\documentclass{article}'
    print '\\usepackage{pgfplots}'
    print '\\begin{document}'

    #TESTS GO HERE

    
    plot_tests({'Berno':(50,1.0,1,0,0),
                'Gauss':(50,1.0,1,0,0)},
                axis='semilogyaxis',trial_range=[0],bvsg=True,
               title="Comparison of Bernoulli and Gaussian")
    
    plot_tests({'Berno':(50,1.0,0,0,0),
                'Berno+IW':(50,1.0,0,0,1)},
                axis='semilogyaxis',trial_range=range(2),
               title="Comparison of Vanilla and Importance Weighted Cost")
    plot_tests({'Berno':(50,1.0,0,0,0),
                'Berno+IW':(50,1.0,0,0,1)},
                plot='covar_cut',ylabel='Active Dimensions',trial_range=range(2),
               title="Comparison of Vanilla and Importance Weighted Activity")
    params={}
    for keep in [1.0,0.9,0.7,0.5]:
        params['KP='+str(keep)]=(50,keep,0,0,1)
    plot_tests(params,plot='cost',trial_range=range(1),
    title="Comparison of Dropout Values for Cost")
    plot_tests(params,plot='covar_cut',ylabel='Active Dimensions',trial_range=range(2),
    title="Comparison of Dropout Values for Activity")
    shade_tests(params,trial_range=range(1))
    plot_tests({'Berno+IW':(50,1.0,0,0,1),
                'Berno+IW+BN':(50,1.0,1,0,1)},
               axis='semilogyaxis',trial_range=range(2),
               title="Comparison of Importance Weighted and Batch Normalized for Cost")
    plot_tests({'Berno+IW':(50,1.0,0,0,1),
                'Berno+IW+WU':(50,1.0,0,1,1)},
               axis='semilogyaxis',trial_range=range(2),
               title="Comparison of Importance Weighted and Warmed-Up for Cost")
    plot_tests({'Berno+IW':(50,1.0,0,0,1),
                'Berno+IW+BN':(50,1.0,1,0,1)},
               plot='covar_cut',ylabel='Active Dimensions',trial_range=range(2),
               title="Comparison of Importance Weighted and Batch Normalized for Activity")
    plot_tests({'Berno+IW':(50,1.0,0,0,1),
                'Berno+IW+WU':(50,1.0,0,1,1)},
               plot='covar_cut',ylabel='Active Dimensions',trial_range=range(2),
               title="Comparison of Importance Weighted and Warmed-Up for Activity")
    params={}
    for dim in [2, 10, 20, 50, 100]:
        params['D='+str(dim)]=(dim,1.0,0,0,1)
    plot_tests(params,plot='cost',trial_range=range(1),
    title="Comparison of Dimension Values for Cost")
    plot_tests(params,plot='covar_cut',ylabel='Active Dimensions',trial_range=range(1),
    title="Comparison of Dimension Values for Activity")
    #shade_tests(params,trial_range=range(1))
    shade_tests({'Berno':(50,1.0,0,0,0),
                'Berno+IW':(50,1.0,0,0,1),
                'Berno+BN+IW':(50,1.0,1,0,1),
                 'Berno+WU+IW':(50,1.0,1,0,1),
                'Berno+BN+WU+IW':(50,1.0,1,1,1)},
                trial_range=[0])
    
    # plot_tests({'Gauss':(50,1.0,'gaussian',0,0),
    #             'Berno':(50,1.0,'bernoulli',0,0)})
    # plot_tests({'Berno':(50,1.0,'bernoulli',0,0),
    #             'Berno+BN':(50,1.0,'bernoulli',1,0),
    #             'Berno+BN+WU':(50,1.0,'bernoulli',1,1)},
    #            axis='semilogyaxis',trial_range=[1])
    # plot_tests({'Gauss':(50,1.0,'gaussian',0,0),
    #             'Berno':(50,1.0,'bernoulli',0,0)},
    #            plot='covar_cut',label='Active Dimensions')
    # plot_tests({'Berno':(50,1.0,'bernoulli',0,0),
    #             'Berno+BN':(50,1`.0,'bernoulli',1,0),
    #             'Berno+BN+WU':(50,1.0,'bernoulli',1,1)},
    #            plot='covar_cut',ylabel='Active Dimensions',title='Active Dimensions for Parameters')
    #shade_tests({'Gauss':(50,1.0,'gaussian',0,0),
    #             'Berno':(50,1.0,'bernoulli',0,0)})
    # shade_tests({'Berno':(50,1.0,'bernoulli',0,0),
    #             'Berno+BN':(50,1.0,'bernoulli',1,0),
    #             'Berno+BN+WU':(50,1.0,'bernoulli',1,1)},
    #             trial_range=[1])
    # plot_tests({'Berno':(50,1.0,0,0,0),
    #             'Berno+IW':(50,1.0,0,0,1),
    #             'Berno+BN+IW':(50,1.0,1,0,1),
    #             'Berno+BN+WU+IW':(50,1.0,1,1,1)},
    #            axis='semilogyaxis',trial_range=[0])
    # shade_tests({'Berno':(50,1.0,0,0,0),
    #             'Berno+IW':(50,1.0,0,0,1),
    #             'Berno+BN+IW':(50,1.0,1,0,1),
    #             'Berno+BN+WU+IW':(50,1.0,1,1,1)},
    #             trial_range=[0])
    #Test dropout rate
    # params={}
    # for keep in [1.0,0.9,0.8,0.7,0.6]:
    #     params['KP='+str(keep)]=(50,keep,1,1,1)
    # plot_tests(params,plot='cost',trial_range=range(4))
    # plot_tests(params,plot='covar_cut',ylabel='Active Dimensions',title='Active Dimensions for Parameters',trial_range=range(4))
    # shade_tests(params,trial_range=range(4))
    #Test initial latent dimensions
    # params={}
    # for dim in [2, 10, 20, 50, 100]:
    #     params['D='+str(dim)]=(dim,1.0,0,0,1)
    # plot_tests(params,plot='cost',trial_range=range(1))
    # plot_tests(params,plot='covar_cut',ylabel='Active Dimensions',title='Active Dimensions for Parameters',trial_range=range(1))
    # shade_tests(params,trial_range=range(1))
    #LATEX FOOTER
    print '\\end{document}'

    #Cleaning up the hack
    sys.stdout.flush()
    sys.stdout.close()
    sys.stdout=stdout_hold
    print "Done!"
