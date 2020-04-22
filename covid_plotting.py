from covid_functions import *
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

#Load data from JH repository
base_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
cases_global = format_JH(base_url+'time_series_covid19_confirmed_global.csv',['Lat','Long'],['Country/Region','Province/State'])
deaths_global = format_JH(base_url+'time_series_covid19_deaths_global.csv',['Lat','Long'],['Country/Region','Province/State'])
cases_US = format_JH(base_url+'time_series_covid19_confirmed_US.csv',['UID','iso2','iso3','code3','FIPS','Lat','Long_','Combined_Key'],['Country_Region','Province_State','Admin2'])
deaths_US = format_JH(base_url+'time_series_covid19_deaths_US.csv',['UID','iso2','iso3','code3','FIPS','Lat','Long_','Combined_Key','Population'],['Country_Region','Province_State','Admin2'])
cases_US = cases_US.T.groupby(level=[0,1]).sum().T
deaths_US = deaths_US.T.groupby(level=[0,1]).sum().T
#Join US and global data into single table
cases = cases_global.join(cases_US)
deaths = deaths_global.join(deaths_US)

#Load full prediction tables, with confidence bounds
pred_date_apr15 = datetime(2020,4,15)
predictions_deaths_apr15 = format_predictions('output/predictions_deaths_apr15.csv')
predictions_cases_apr15 = format_predictions('output/predictions_cases_apr15.csv')

#Get best-fit th at higher resolution
params_deaths_apr15=pd.read_csv('output/params_deaths_apr15.csv').fillna(value='NaN').set_index(['Country/Region','Province/State'])
params_cases_apr15=pd.read_csv('output/params_cases_apr15.csv').fillna(value='NaN').set_index(['Country/Region','Province/State'])
predictions_deaths_apr15['th'] = tref + pd.to_timedelta(params_deaths_apr15['th'],unit='days')
predictions_cases_apr15['th'] = tref + pd.to_timedelta(params_cases_apr15['th'],unit='days')

#Get China parameters from time near peak
params_china = fit_all(deaths.loc[:datetime(2020,2,25)],plot=False,p0=50).loc['China','Hubei']

def plot_region(country,region,forecast_days=20,thresh=10,ax=None,log_scale=False,new_predictions=None):

	#Load predictions
	if new_predictions is None:
		predictions_deaths = predictions_deaths_apr15
		predictions_cases = predictions_cases_apr15
		pred_date = pred_date_apr15
	else:
		pred_date = deaths.index[-1]
		try:
			predictions_deaths,predictions_cases = new_predictions
		except:
			print('new_predictions must be a tuple of two Pandas dataframes with predictions for deaths and cases')
			return np.nan

	#Set up axes
	if ax is None:
		fig,ax=plt.subplots(2,figsize=(12,12),sharex=True)
		fig.subplots_adjust(hspace=0.05)

	#Set up prediction time axis
	t0 = cases[country,region].loc[cases[country,region]>thresh].index[0]
	daymax = int((datetime.today()+timedelta(days=forecast_days)-t0)/timedelta(days=1))
	t = pd.to_datetime([t0+timedelta(days=k) for k in range(0,daymax)])
	if log_scale:
		time_axis_pred = (t-t0+timedelta(days=1))/timedelta(days=1)
		pred_date = (pred_date-t0+timedelta(days=1))/timedelta(days=1)
	else:
		time_axis_pred = t

	######FATALITIES########
	#Load data
	data = deaths[country,region].copy()
	data = data.loc[data>thresh]

	#Set up data time axis
	if log_scale:
		time_axis_data = (data.index-t0+timedelta(days=1))/timedelta(days=1)
	else:
		time_axis_data = data.index

	#Plot data
	ax[0].plot(time_axis_data,data.values,marker='o',label='data')

	#Plot predictions
	if (country,region) in predictions_deaths.index.tolist():
	    pred = predictions_deaths.loc[country,region]
	    tau = ((t-pred['th'])/timedelta(days=1))/pred['sigma']
	    tau_low = ((t-pred['th_low'])/timedelta(days=1))/pred['sigma_low']
	    tau_high = ((t-pred['th_high'])/timedelta(days=1))/pred['sigma_high']
	    ax[0].fill_between(time_axis_pred,pred['Nmax_low']*norm.cdf(tau_low),pred['Nmax_high']*norm.cdf(tau_high),color='gray',alpha=0.5)
	    ax[0].plot(time_axis_pred,pred['Nmax']*norm.cdf(tau),label='prediction')
	    ax[0].plot([pred_date,pred_date],[data.min(),3*data.max()],'k--',label='prediction date')
	    ax[0].legend()
	else:
	    ax[0].text(0.05,0.88,'No prediction available',fontsize=11,transform=ax[0].transAxes)
	ax[0].set_yscale('log')
	ax[0].set_title(', '.join([country,region]))
	ax[0].set_ylim((thresh,None))
	ax[0].set_ylabel('Cumulative fatalities')

	############CASES###############
	#Load data
	data = cases[country,region].copy()
	data = data.loc[data>thresh]

	#Set up data time axis
	if log_scale:
		time_axis_data = (data.index-t0+timedelta(days=1))/timedelta(days=1)
	else:
		time_axis_data = data.index

	#Plot data
	ax[1].plot(time_axis_data,data.values,marker='o',label='data')

	#Plot predictions
	if (country,region) in predictions_cases.index.tolist():
	    pred = predictions_cases.loc[country,region]
	    tau = ((t-pred['th'])/timedelta(days=1))/pred['sigma']
	    tau_low = ((t-pred['th_low'])/timedelta(days=1))/pred['sigma_low']
	    tau_high = ((t-pred['th_high'])/timedelta(days=1))/pred['sigma_high']
	    ax[1].fill_between(time_axis_pred,pred['Nmax_low']*norm.cdf(tau_low),pred['Nmax_high']*norm.cdf(tau_high),color='gray',alpha=0.5)
	    ax[1].plot(time_axis_pred,pred['Nmax']*norm.cdf(tau),label='prediction')
	    ax[1].plot([pred_date,pred_date],[data.min(),3*data.max()],'k--',label='prediction date')
	else:
	    ax[1].text(0.05,0.88,'No prediction available',fontsize=11,transform=ax[1].transAxes)
	ax[1].set_yscale('log')
	ax[1].set_ylabel('Cumulative confirmed cases')
	ax[1].set_ylim((thresh,None))
	if log_scale:
		ax[1].set_xscale('log')
		ax[1].set_xlabel('Elapsed time (days)')
	else:
		ax[1].xaxis.set_minor_locator(mdates.DayLocator())
		ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
		ax[1].set_xlabel('Date (mm-dd)')
		fig.autofmt_xdate()
	plt.show()

def plot_collapse(params_deaths,params_cases,thresh=500):
	#Extract countries with current fatalities above threshold
	thresh = 500
	current_fatalities = deaths.iloc[-1]
	current_fatalities = current_fatalities.sort_index().drop('US').loc[current_fatalities>thresh].sort_values(ascending=False)
	top_countries = current_fatalities.index
	colors = (sns.color_palette()+sns.color_palette('pastel')+sns.color_palette('dark'))*10
    
	#Fatalities rescaled
	fig,ax = data_collapse(deaths,params_deaths.loc[top_countries],endpoint=True,ms=5,colors=colors)
	#Get legend plotted correctly
	fig.set_figheight(7)
	fig.set_figwidth(12)
	ax.set_ylim((1e-3,1.5))
	ax.set_xlim((-3,3))
	ax.set_ylabel(r'Relative fatalities $N/N_{\rm max}$')
	ax.set_xlabel(r'Rescaled time $(t-t_h)/\sigma$')
	ax.set_position([0.22,0.22,0.7,0.7])
	ax.legend(loc='upper right',bbox_to_anchor=(1.25, 1))
	ax.set_title('Global Fatalities')
	plt.show()
    
	#Cases rescaled
	fig,ax = data_collapse(cases,params_cases.loc[top_countries],endpoint=True,ms=5,colors=colors)
	#Get legend plotted correctly
	fig.set_figheight(7)
	fig.set_figwidth(12)
	ax.set_ylim((1e-3,1.5))
	ax.set_xlim((-3,3))
	ax.set_ylabel(r'Relative confirmed cases $N/N_{\rm max}$')
	ax.set_xlabel(r'Rescaled time $(t-t_h)/\sigma$')
	ax.set_position([0.22,0.22,0.7,0.7])
	ax.legend(loc='upper right',bbox_to_anchor=(1.25, 1))
	ax.set_title('Global Confirmed Cases')
	plt.show()

	###########################################
	#Extract states with current fatalities above threshold
	current_fatalities = deaths['US'].iloc[-1]
	current_fatalities = current_fatalities.loc[current_fatalities>thresh].sort_index().drop('NaN').sort_values(ascending=False)
	top_countries = current_fatalities.index

	#Fatalities rescaled
	fig,ax = data_collapse(deaths['US'],params_deaths.loc['US'].loc[top_countries],endpoint=True,ms=5,colors=colors)
	#Get legend plotted correctly
	fig.set_figheight(7)
	fig.set_figwidth(12)
	ax.set_ylim((1e-3,1.5))
	ax.set_xlim((-3,3))
	ax.set_ylabel(r'Relative fatalities $N/N_{\rm max}$')
	ax.set_xlabel(r'Rescaled time $(t-t_h)/\sigma$')
	ax.set_position([0.22,0.22,0.7,0.7])
	ax.legend(loc='upper right',bbox_to_anchor=(1.25, 1))
	ax.set_title('US Fatalities')
	plt.show()
    
	#Cases rescaled
	fig,ax = data_collapse(cases['US'],params_cases.loc['US'].loc[top_countries],endpoint=True,ms=5,colors=colors)
	#Get legend plotted correctly
	fig.set_figheight(7)
	fig.set_figwidth(12)
	ax.set_ylim((1e-3,1.5))
	ax.set_xlim((-3,3))
	ax.set_ylabel(r'Relative confirmed cases $N/N_{\rm max}$')
	ax.set_xlabel(r'Rescaled time $(t-t_h)/\sigma$')
	ax.set_position([0.22,0.22,0.7,0.7])
	ax.legend(loc='upper right',bbox_to_anchor=(1.25, 1))
	ax.set_title('US Confirmed Cases')
	plt.show()