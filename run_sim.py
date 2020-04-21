from covid_functions import *

#Make graphs
N=5000  #Number of nodes
graphs = {'gamma_2.5':nx.expected_degree_graph(nx.utils.powerlaw_sequence(N, 2.5), selfloops=False),
			'gamma_3.1':nx.expected_degree_graph(nx.utils.powerlaw_sequence(N, 3.1), selfloops=False),
          	'gamma_3.5':nx.expected_degree_graph(nx.utils.powerlaw_sequence(N, 3.5), selfloops=False),
          	'gamma_4':nx.expected_degree_graph(nx.utils.powerlaw_sequence(N, 4), selfloops=False),
          	'ER_1.5':nx.gnp_random_graph(N, 1.5/N),
          	'ER_2':nx.gnp_random_graph(N, 2/N),
          	'ER_6':nx.gnp_random_graph(N, 6/N)}

#Set up waiting time parameters
muG = 50 #Mean waiting time
sigG_list = {'Markovian':muG,'Non-Markovian':muG/2}

tmax = 1000
time_axis = [(datetime.today()-timedelta(days=tmax-k)).isoformat()[:10] for k in range(tmax)]
sim_data = pd.DataFrame(index=time_axis,columns=pd.MultiIndex.from_tuples([(item,name) for item in graphs.keys() for name in sigG_list.keys()]))

for item in sim_data.keys():
	print(item)
	t,cum_cases = simulate_pandemic_edges(graphs[item[0]],muG,sigG_list[item[1]],N_0=5,p=1,tmax=tmax,sampling='Gamma')
	sim_data[item] = cum_cases
	sim_data.to_csv('output/simulation_exp.csv')