import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy.special import gamma,gammainc,gammaincc
from scipy.stats import norm
from scipy.optimize import minimize,root_scalar
import networkx as nx
from operator import itemgetter


ep = 1e-80
tref = pd.to_datetime('2020-01-01')

def format_JH(url,drop_list,columns):
    data = pd.read_csv(url)
    if len(columns) == 2:
        data[columns[1]] = data[columns[1]].fillna(value='NaN')
    data = data.T.drop(drop_list).T.set_index(columns).T
    data.index = pd.to_datetime(data.index,format='%m/%d/%y')
    
    return data

def format_kaggle(folder,metric):
    data_full = pd.read_csv(folder+'train.csv')
    data = data_full.pivot_table(index='Date',columns=['Country_Region','Province_State'],values=metric)
    data.index = pd.to_datetime(data.index,format='%Y-%m-%d')
    
    return data

def format_predictions(path):
    pred = pd.read_csv(path).fillna(value='NaN').set_index(['Country/Region','Province/State'])
    for item in ['Nmax','Nmax_low','Nmax_high','sigma','sigma_low','sigma_high']:
        pred[item] = pd.to_numeric(pred[item])
    for item in ['th','th_low','th_high']:
        pred[item] = pd.to_datetime(pred[item],format='%Y-%m-%d')
    return pred

def load_sim(path):
    data = pd.read_csv(path,index_col=0,header=[0,1])
    data.index = pd.to_datetime(data.index,format='%Y-%m-%d')
    for item in data.keys():
        data[item] = pd.to_numeric(data[item])
    
    return data

def cbarr(t):
    return 1/(np.sqrt(2*np.pi)*(1-norm.cdf(t)+ep))

def tmean(tf,params):
    th,sigma = params
    tau = (tf-th)/sigma
    
    return -sigma*cbarr(-tau)*np.exp(-tau**2/2)+th

def tvar(tf,params):
    th,sigma = params
    tau = (tf-th)/sigma
    
    return sigma**2*cbarr(-tau)*(np.sqrt(np.pi/2)*(1+np.sign(tau)*gammaincc(3/2,tau**2/2))-cbarr(-tau)*np.exp(-tau**2/2))

def cost_init(params,data,tf):
    th,sigma = params
    tmean_sample = (data.index.values*data.values).sum()/data.values.sum()
    tvar_sample = (((data.index.values-tmean_sample)**2)*data.values).sum()/data.values.sum()
    
    return (tmean_sample-tmean(tf,params))**2 + (tvar_sample-tvar(tf,params))**2

def cost_p(params,data,prior):
    th,logK,sigma = params
    t = data.index.values
    tau = (t-th)/sigma
    if prior is not None:
        mean_sigma, var_sigma = prior
        penalty = (sigma-mean_sigma)**2/(2*var_sigma)
    else:
        penalty = 0
    prediction = logK+np.log((norm.cdf(tau)+ep))
    
    return ((np.log(data.values)-prediction)**2).sum()/2 + penalty

def jac_p(params,data,prior):
    th,logK,sigma = params
    t = data.index.values
    tau = (t-th)/sigma
    if prior is not None:
        mean_sigma, var_sigma = prior
        dpenalty = (sigma-mean_sigma)/var_sigma
    else:
        dpenalty = 0

    prediction = logK+np.log((norm.cdf(tau)+ep))
    err = np.log(data.values)-prediction
    dlogNdt = np.exp(-tau**2/2)/(np.sqrt(2*np.pi)*sigma*(norm.cdf(tau)+ep))
    
    return np.asarray([(dlogNdt*err).sum(),-err.sum(),(tau*dlogNdt*err).sum()])+np.asarray([0,0,dpenalty])

def hess_p(params,data,prior):
    th,logK,sigma = params
    t = data.index.values
    tau = (t-th)/sigma
    if prior is not None:
        mean_sigma, var_sigma = prior
        d2penalty = 1/var_sigma
    else:
        d2penalty = 0

    prediction = logK+np.log((norm.cdf(tau)+ep))
    err = np.log(data.values)-prediction
    dlogNdt_s = np.exp(-tau**2/2)/(np.sqrt(2*np.pi)*(norm.cdf(tau)+ep))
    dlogNdth = -dlogNdt_s/sigma
    dlogNdlogK = np.ones(len(t))
    dlogNdsig = -tau*dlogNdt_s/sigma
    d2Ndth2_N = -tau*dlogNdt_s/sigma**2
    d2Ndsig2_N = 2*tau*(1-tau**2/2)*dlogNdt_s/(sigma**2)
    d2Ndsigdth_N = (1-2*tau**2/2)*dlogNdt_s/sigma**2

    term1 = np.asarray([[((-d2Ndth2_N+dlogNdth**2)*err).sum(), 0, ((-d2Ndsigdth_N+dlogNdth*dlogNdsig)*err).sum()],
                         [0, 0, 0],
                         [((-d2Ndsigdth_N+dlogNdth*dlogNdsig)*err).sum(), 0, ((-d2Ndsig2_N+dlogNdsig**2)*err).sum()]])
    term2 = np.asarray([[(dlogNdth**2).sum(), (dlogNdth*dlogNdlogK).sum(), (dlogNdth*dlogNdsig).sum()],
                         [(dlogNdth*dlogNdlogK).sum(), (dlogNdlogK**2).sum(), (dlogNdsig*dlogNdlogK).sum()],
                         [(dlogNdth*dlogNdsig).sum(), (dlogNdsig*dlogNdlogK).sum(), (dlogNdsig**2).sum()]])
    
    term3 = np.zeros((3,3))
    term3[2,2] = d2penalty

    return term1 + term2+ term3

def th_err(th,data,sigma,tf):
    
    tmean_sample = (data.index.values*data.values).sum()/data.values.sum()
    tau = (tf-th)/sigma
    tmean = -sigma*cbarr(-tau)*np.exp(-tau**2/2)+th
    
    return tmean_sample-tmean

def cost_p_sig(params,data,sigma):
    th,logK = params
    t = data.index.values
    tau = (t-th)/sigma
    prediction = logK+np.log((norm.cdf(tau)+ep))
    
    return 0.5*((np.log(data.values)-prediction)**2).sum()

def jac_p_sig(params,data,sigma):
    th,logK = params
    t = data.index.values
    tau = (t-th)/sigma

    prediction = logK+np.log((norm.cdf(tau)+ep))
    err = np.log(data.values)-prediction
    dlogNdt = np.exp(-tau**2/2)/(np.sqrt(np.pi*2)*sigma*(norm.cdf(tau)+ep))
    
    return np.asarray([(dlogNdt*err).sum(),
                       -err.sum()])

def fit_erf_sig(data,p0=5e2,sigma=7):
    
    #Get initial conditions
    train = data.loc[data>0].diff().iloc[1:]
    t = (train.index-tref)/timedelta(days=1)
    train.index = t
    train = pd.to_numeric(train)
    th0 = (t.values*train.values).sum()/train.values.sum()
    out = root_scalar(th_err,args=(train,sigma,t[-1]),x0=th0,x1=th0+10)
    th0 = out.root
    tau0 = (t[-1]-th0)/sigma
    logK0 = np.log(data.iloc[-1]/(norm.cdf(tau0)+ep))
    params = [th0,logK0,sigma]

    #Train the model
    train = data.loc[data>p0]
    t = (train.index-tref)/timedelta(days=1)
    train.index = t
    train = pd.to_numeric(train)
    out = minimize(cost_p_sig,[th0,logK0],args=(train,sigma),jac=jac_p_sig,method='BFGS')
    params = list(out.x)+[sigma,2*out.fun/len(train)]
    
    return params


def fit_erf(data,p0=5e2,verbose=False,prior=None):

    #Get initial conditions
    train = data.loc[data>0].diff().iloc[1:]
    t = (train.index-tref)/timedelta(days=1)
    train.index = t
    train = pd.to_numeric(train)
    th0 = (t.values*train.values).sum()/train.values.sum()
    sig0 = np.sqrt(((t-th0).values**2*train.values).sum()/train.values.sum())
    tf = t[-1]
    if prior is not None:
        mean_sigma, var_sigma = prior
        lb = mean_sigma-2*np.sqrt(var_sigma)
        ub = mean_sigma+2*np.sqrt(var_sigma)
    else:
        lb = None
        ub = None
    out = minimize(cost_init,[th0,sig0],args=(train,tf),bounds=((None,None),(lb,ub)))
    th0,sig0 = out.x
    tau0 = (tf-th0)/sig0
    logK0 = np.log(data.iloc[-1]/(norm.cdf(tau0)+ep))
    
    #Fit the curve
    train = data.loc[data>p0]
    t = (train.index-tref)/timedelta(days=1)
    train.index = t
    train = pd.to_numeric(train)
    out = minimize(cost_p,[th0,logK0,sig0],args=(train,prior),method='Nelder-Mead')
    #out = minimize(cost_p,[th0,logK0,sig0],args=(train,prior),jac=jac_p,hess=hess_p)
    #if not out.success:
    #    out = minimize(cost_p,[th0,logK0,sig0],args=(train,prior),method='Nelder-Mead')
        #out = minimize(cost_p,out.x,args=(train,),jac=jac_p,method='BFGS')
    params = list(out.x)+[2*out.fun/len(train)]
    if verbose:
        print(out)
    
    return params, [th0,logK0,sig0], out.success

def fit_all(data,p0=5e2,plot=False,ylabel=None,prior=None):
    params_list = pd.DataFrame(index=data.keys(),columns=['th','logK','sigma','score'])
    for item in data.keys():
        params_list.loc[item] = [np.nan,np.nan,np.nan,np.nan]
        if (data[item].diff()>1).sum() > 7:
            if (data[item]>p0).sum() > 5:
                params,params_0,success = fit_erf(data[item],p0=p0,prior=prior)
                params_list.loc[item] = params

                if plot:
                    fig,ax,params_good = plot_predictions(data[item],params)
                    ax.set_title(item)
                    ax.set_ylabel(ylabel)
                    ax.set_ylim((10,None))
                    plt.show()
            
    return params_list

def predict_all(data,params_list,p0=50,c=0.95):
    pred_idx = params_list.index.copy()
    predictions = []
    for item in pred_idx:
        print(item[0]+', '+item[1])
        train = data[item]
        params = params_list.loc[item].copy()
        try:
            params_sweep = sweep_sigma(params,train,p0)
            sigma,prob,scoremax = get_score_thresh(params_sweep,len(train.loc[train>p0]),c)
            params_good = params_sweep[params_sweep[:,3]<=scoremax]
            total = np.exp(params_good[:,1])
            th = [pd.Timestamp.isoformat((tref+pd.to_timedelta(params_good[:,0],unit='days'))[k])[:10] for k in range(len(params_good))]
            sigma = params_good[:,2]
            best = np.argmin(params_good[:,-1])
            predictions.append([total[best],total[0],total[-1],sigma[best],sigma[0],sigma[-1],th[best],th[0],th[-1]])
        except:
            print('---------------Failed---------------')
            pred_idx = pred_idx.drop(item)
    predictions = pd.DataFrame(predictions,index=pred_idx,columns=['Nmax','Nmax_low','Nmax_high','sigma','sigma_low','sigma_high','th','th_low','th_high'])
    return predictions

def data_collapse(data,params,scale=True,colors=list(sns.color_palette())*10,ax=None,ms=10,
                  endpoint=False,alpha=1,labels=True):
    if ax is None:
        fig,ax=plt.subplots(figsize=(4,3))
        fig.subplots_adjust(left=0.22,bottom=0.22,right=0.9)
    else:
        fig = np.nan
    k = 0
    for item in params.index:
        th,logK,sigma = params[['th','logK','sigma']].loc[item]
        if th is not 'NaN':
            data_plot = data[item].copy()
            if scale:
                data_plot.index = ((data_plot.index-tref)/pd.to_timedelta(1,unit='days') - th)/sigma
                data_plot = data_plot/np.exp(logK)
            else:
                data_plot.index = (data_plot.index-tref)/pd.to_timedelta(1,unit='days')
            if labels:
                if np.shape(item) is ():
                    label = item
                elif item[0] in ['China','US']:
                    label = ', '.join([item[0],item[1]])
                else:
                    label = item[0]
            else:
                label=None
            ax.semilogy(data_plot.index,data_plot.values,label=label,color=colors[k],alpha=alpha)
            if endpoint:
                ax.semilogy([data_plot.index[-1]],[data_plot.values[-1]],'o',color=colors[k],markersize=ms)
            k+=1
        else:
            print('----------------')
            print(', '.join(item)+' not included.')
            
    return fig,ax


def make_prior(data,params,thresh,plot=False,buffer=0):

    params_valid = params.loc[data.iloc[-1]>thresh].replace('NaN',np.nan).dropna().sort_values('sigma')
    not_peaked = params_valid['th']>(data.index[-1]-tref+pd.to_timedelta(buffer,unit='days'))/pd.to_timedelta(1,unit='days')
    peaked = params_valid['th']<=(data.index[-1]-tref+pd.to_timedelta(buffer,unit='days'))/pd.to_timedelta(1,unit='days')
    params_valid = params_valid.loc[peaked]
    
    if plot:
        params_valid['sigma'].loc[peaked].plot.hist()
        
    peaked = peaked.loc[peaked].index.tolist()
    not_peaked = not_peaked.loc[not_peaked].index.tolist()
    
    return params_valid['sigma'].loc[peaked].mean(), params_valid['sigma'].loc[peaked].var(), peaked, not_peaked

def conf_bounds(t,params,hess_inv):
    th,logK,sigma,score = params
    
    lb = []
    ub = []
    ml = []
    for ti in t:
        tau = (ti-th)/sigma
        prediction = logK+np.log((norm.cdf(tau)+ep))
        dlogNdt = np.exp(-tau**2/2)/(np.sqrt(2*np.pi)*sigma*(norm.cdf(tau)+ep))
        dlogNdx = np.asarray([-dlogNdt,1,-tau*dlogNdt])
        sigma_pred2 = dlogNdx[np.newaxis,:].dot(hess_inv.dot(dlogNdx)).squeeze()*score
        ub.append(np.exp(prediction+2*np.sqrt(sigma_pred2)))
        lb.append(np.exp(prediction-2*np.sqrt(sigma_pred2)))
        ml.append(np.exp(prediction))
    return np.asarray(lb), np.asarray(ml), np.asarray(ub)

def conf_bounds_eig(t,params,hess_inv):
    th,logK,sigma,score = params
    v,u = np.linalg.eig(hess_inv*score)
    sloppy_v = v[0]
    sloppy_u = u[:,0]
    params_upper = params[:3]+2*sloppy_u*np.sqrt(sloppy_v)
    params_lower = params[:3]-2*sloppy_u*np.sqrt(sloppy_v)
    
    tau = (t-th)/sigma
    ml = np.exp(logK)*(norm.cdf(tau)+ep)
    
    th,logK,sigma = params_lower
    tau = (t-th)/sigma
    lb = np.exp(logK)*(norm.cdf(tau)+ep)
    
    th,logK,sigma = params_upper
    tau = (t-th)/sigma
    ub = np.exp(logK)*(norm.cdf(tau)+ep)
    
    return lb,ml,ub

def get_sigvar(params,data,p0):
    th,logK,sigma0,score0 = params
    train = pd.to_numeric(data.loc[data>p0])
    train.index=(train.index-tref)/timedelta(days=1)
    H = hess_p(params[:-1],train,None)
    return np.linalg.inv(H)[2,2]*params[-1]


def sweep_sigma(params,data,p0,sig_bound=30):
    th,logK,sigma0,score0 = params
    sigvar = get_sigvar(params,data,p0)
    if sigvar < 0:
        sigvar = 100
    
    params_sweep = []
    for sigma in np.logspace(np.log10(np.max([sigma0-4*np.sqrt(sigvar),1])),np.log10(sigma0+sig_bound*np.sqrt(sigvar)),200):
        params_sweep.append(fit_erf_sig(data,sigma=sigma,p0=p0))
    return np.asarray(params_sweep)

def get_score_thresh(params_sweep,M,c):
    sigma = params_sweep[:,2]
    dsig = np.diff(sigma)
    sigma = sigma[1:]
    score = params_sweep[1:,3]
    sig_xi2 = np.min(score)
    prob = np.exp(-score*M/(2*sig_xi2))/(np.exp(-score*M/(2*sig_xi2))*dsig).sum()
    
    score_set = list(set(score))
    score_set.sort()
    score_set = np.asarray(score_set)
    pcum = np.asarray([(prob[score<=val]*dsig[score<=val]).sum() for val in score_set])
    scoremax = score_set[pcum<=c][-1]
    
    return sigma, prob, scoremax

def conf_bounds_sigma(t,params_sweep,M,c):

    sigma,prob,scoremax = get_score_thresh(params_sweep,M,c)
    params_good = params_sweep[params_sweep[:,3]<=scoremax]
    
    th,logK,sigma = params_good[np.argmin(params_good[:,-1]),:3]
    tau = (t-th)/sigma
    ml = np.exp(logK)*(norm.cdf(tau)+ep)
    
    th,logK,sigma = params_good[0,:3]
    tau = (t-th)/sigma
    lb = np.exp(logK)*(norm.cdf(tau)+ep)
    
    th,logK,sigma = params_good[-1,:3]
    tau = (t-th)/sigma
    ub = np.exp(logK)*(norm.cdf(tau)+ep)
    
    return lb,ml,ub,params_good

def plot_predictions(data,params,t_pred = None,conf_type=None,p0=5e2,log_scale=False,c=0.95,
                     start_cutoff=5,prior=None,mask=None,th_true=None,ax=None,sig_bound=80):
    colors = sns.color_palette()
    th,logK,sigma,score = params
    plot_data = data.loc[data>=start_cutoff].iloc[1:]

    #Set up time axis
    t = (plot_data.index-tref)/timedelta(days=1)
    if th_true is None:
        th_true = th
    if t_pred is None:
        t_pred = t
    else:
        t_pred = (t_pred-tref)/timedelta(days=1)
    if log_scale:
        t0 = (data.loc[data>=start_cutoff].index[0]-tref)/timedelta(days=1)
        t_axis = t_pred-t0
    else:
        t_axis = t_pred-th_true

    #Set up figure
    if ax is None:
        fig,ax=plt.subplots(figsize=(4,3))
        fig.subplots_adjust(left=0.22,bottom=0.22,right=0.9)
    else:
        fig = np.nan

    #Plot the data
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlabel('Elapsed time (days)')
        ax.plot(t-t0,plot_data.values,'o',color=colors[0],markersize=4,label='Data')
    else:
        ax.set_xlabel('Time after peak (days)')
        ax.plot(t-th_true,plot_data.values,'o',color=colors[0],markersize=4,label='Data')

    #Plot fit
    tau = (t_pred-th)/sigma
    pred = np.exp(logK)*(norm.cdf(tau)+ep)
    if conf_type is None:
        ax.plot(t_axis,pred,color=colors[1],markersize=4,label='Fit')
    #Plot predictions with confidence interval
    else:
        train = data.loc[data>p0]
        if mask is not None:
            train = train.iloc[:-mask]
        if conf_type=='eig':
            train.index = (train.index-tref)/timedelta(days=1)
            train = pd.to_numeric(train)
            hess_inv = np.linalg.inv(hess_p([th,logK,sigma],train,prior))
            lb,ml,ub = conf_bounds_eig(t_pred,params,hess_inv)
        elif conf_type=='LCA':
            train.index = (train.index-tref)/timedelta(days=1)
            train = pd.to_numeric(train)
            hess_inv = np.linalg.inv(hess_p([th,logK,sigma],train,prior))
            lb,ml,ub = conf_bounds(t_pred,params,hess_inv)
        elif conf_type=='sigma':
            M = len(train)
            params_sweep = sweep_sigma(params,train,0,sig_bound=sig_bound)
            lb,ml,ub,params_good = conf_bounds_sigma(t_pred,params_sweep,M,c)
        ax.fill_between(t_axis,lb,ub,alpha=0.5,color='gray')
        ax.plot(t_axis,ub,color='k',lw=0.5)
        ax.plot(t_axis,lb,color='k',lw=0.5)
        ax.plot(t_axis,pred,color=colors[1],label='Fit')
    #Set up y axis
    ax.set_yscale('log')
    ax.set_ylim((10,None))
    if conf_type != 'sigma':
        params_good = np.nan
    return fig,ax,params_good

def simulate_pandemic_nodes(G,muG,sigG,sampling='Gaussian',N_0=5,p=1,tmax=60):
    
    #Sample waiting times
    N = G.number_of_nodes()
    if sampling == 'Gaussian':
        graph_waiting_times=np.abs(np.random.normal(muG, sigG, N))
    elif sampling == 'Exponential':
        graph_waiting_times=np.random.exponential(muG, N)
    elif sampling == 'Gamma':
        theta = sigG**2/muG
        k = muG**2/sigG**2
        graph_waiting_times=np.random.gamma(k,theta,N)
        
    #Create list of what nodes are infected and absolute time at
    #which node infects neighbor node infects all its neighbors
    data=[]
    #This list is of people who have been infected
    infected=[]
    #This is the time at which they will infect their neighbors
    infection_times=[]
    time_infected=[]


    #Draw node to infect
    t=0
    tmax_running = 0
    generation=0
    infected= list(np.random.randint(N,size=N_0))
    infection_times=list(graph_waiting_times[infected])
    infected, infection_times=[list(x) for x in zip(*sorted(zip(infected, infection_times), key=itemgetter(1)))]
    Rtild = []
    t_in = []

    while generation < max(len(infected),1):
        if generation %1000 ==0:
            print('Generation '+str(generation))
        current_node=infected[generation]
        t=infection_times[generation]
        
        #Get neighbors of current node that will infect all
        neighbors=G.neighbors(current_node)
        
        #Find uninfected neighbors
        uninfected_neighbors= list(set(neighbors)-set(infected))
        Rtild.append(len(uninfected_neighbors))
        t_in.append(t)
        #Determine which uninfected neighbors to infect
        infected_neighbors=list(np.array(uninfected_neighbors)[np.random.uniform(size=len(uninfected_neighbors))>1-p])
        #Determine time when infections occur
        neighbor_infection_times=graph_waiting_times[infected_neighbors]+t
        #Update list of infected nodes
        infected=list(infected)+list(infected_neighbors)
        #Update list of infection times
        infection_times=list(infection_times)+list(neighbor_infection_times)
        
        #Repackage
        infected, infection_times=[list(x) for x in zip(*sorted(zip(infected, infection_times), key=itemgetter(1)))]
        time_infected=list(time_infected)+len(uninfected_neighbors)*[t]
        
        generation=generation+1
    
    #Make time axis
    t=np.arange(int(tmax))

    #Extract cumulative number of cases at each time point
    infection_times_array=np.tile(infection_times,(len(t),1))
    t_array=np.tile(t,(len(infection_times),1)).T
    cum_cases=np.sum(infection_times_array < t_array, axis=1)

    return t, cum_cases, t_in, Rtild

def simulate_pandemic_edges(G,muG,sigG,sampling='Gaussian',N_0=5,p=1,tmax=300):
    
    #Make waiting time distribution
    if sampling == 'Gaussian':
        waiting_dist = lambda x: np.abs(random.normal(muG,sigG,x))
    elif sampling == 'Exponential':
        waiting_dist = lambda x: np.random.exponential(muG, x)
    elif sampling == 'Gamma':
        waiting_dist = lambda x: np.random.gamma(muG**2/sigG**2,sigG**2/muG,x)
        

    N = G.number_of_nodes()
    #Draw node to infect
    t=0
    generation=0
    time_infected=list(np.zeros(N_0))
    initially_infected=list(np.random.randint(N,size=N_0))
    infected=initially_infected[:]
    infected_edges=G.edges(infected)
    infection_times=list(waiting_dist(len(infected_edges)))
    infection_edges, infection_times=[list(x) for x in zip(*sorted(zip(infected_edges, infection_times), key=itemgetter(1)))]

    while (generation < max(len(infection_edges),1)):
        #print('Generation '+str(generation)+' Num. infected: '+str(len(infected)))
        if generation %1000 ==0:
            print('Generation '+str(generation))
        current_edge=infection_edges[generation]
        t=infection_times[generation]

        if current_edge[1] not in infected:
            infected_node=current_edge[1]
            #Update global arrays
            infected.append(infected_node)

            time_infected.append(t)
            #print('Infected node'+str(infected_node)+'at time'+str(t))
            #print(infected)
            #Add and sort new edges
            infected_node_neighbors=G.neighbors(infected_node)
            potential_new_edges=G.edges(infected_node_neighbors)
            new_edges=[x for x in potential_new_edges if x[1] not in infected]
            new_infection_times=list(waiting_dist(len(new_edges))+t)
            infection_times=list(infection_times)+list(new_infection_times)
            infected_edges=list(infection_edges)+list(new_edges)
            infection_edges, infection_times=[list(x) for x in zip(*sorted(zip(infected_edges, infection_times), key=itemgetter(1)))]
        generation=generation+1

    #Make time axis
    t=np.arange(int(tmax))

    #Extract cumulative number of cases at each time point
    infection_times_array=np.tile(time_infected,(len(t),1))
    t_array=np.tile(t,(len(time_infected),1)).T
    cum_cases=np.sum(infection_times_array < t_array, axis=1)

    return t, cum_cases