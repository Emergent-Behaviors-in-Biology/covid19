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
pred_date = datetime(2020,4,15)
predictions_deaths = format_predictions('output/predictions_deaths_apr15.csv')
predictions_cases = format_predictions('output/predictions_cases_apr15.csv')

def plot_region(country,region,forecast_days=20,thresh=10,ax=None):

	if ax is None:
		fig,ax=plt.subplots(2,figsize=(10,12),sharex=True)
		fig.subplots_adjust(hspace=0.05)

	t0 = cases[country,region].loc[cases[country,region]>thresh].index[0]

	data = deaths[country,region].copy()
	data = data.loc[data>thresh]
	ax[0].plot(data.index,data.values,marker='o',label='data')
	if (country,region) in predictions_deaths.index.tolist():
	    pred = predictions_deaths.loc[country,region]
	    daymax = int((datetime.today()+timedelta(days=forecast_days)-t0)/timedelta(days=1))
	    t = pd.to_datetime([t0+timedelta(days=k) for k in range(daymax)])
	    tau = ((t-pred['th'])/timedelta(days=1))/pred['sigma']
	    tau_low = ((t-pred['th_low'])/timedelta(days=1))/pred['sigma_low']
	    tau_high = ((t-pred['th_high'])/timedelta(days=1))/pred['sigma_high']
	    ax[0].fill_between(t,pred['Nmax_low']*norm.cdf(tau_low),pred['Nmax_high']*norm.cdf(tau_high),color='gray',alpha=0.5)
	    ax[0].plot(t,pred['Nmax']*norm.cdf(tau),label='prediction')
	    ax[0].legend()
	    ax[0].plot([pred_date,pred_date],[data.min(),3*data.max()],'k--',label='prediction date')
	else:
	    ax[0].text(0.05,0.88,'No prediction available',fontsize=11,transform=ax[0].transAxes)
	ax[0].set_yscale('log')
	ax[0].set_title(', '.join([country,region]))
	ax[0].set_ylim((thresh,None))
	ax[0].set_ylabel('Cumulative fatalities')

	data = cases[country,region].copy()
	data = data.loc[data>thresh]
	ax[1].plot(data.index,data.values,marker='o',label='data')

	if (country,region) in predictions_cases.index.tolist():
	    pred = predictions_cases.loc[country,region]
	    daymax = int((datetime.today()+timedelta(days=forecast_days)-t0)/timedelta(days=1))
	    t = pd.to_datetime([t0+timedelta(days=k) for k in range(daymax)])
	    tau = ((t-pred['th'])/timedelta(days=1))/pred['sigma']
	    tau_low = ((t-pred['th_low'])/timedelta(days=1))/pred['sigma_low']
	    tau_high = ((t-pred['th_high'])/timedelta(days=1))/pred['sigma_high']
	    ax[1].fill_between(t,pred['Nmax_low']*norm.cdf(tau_low),pred['Nmax_high']*norm.cdf(tau_high),color='gray',alpha=0.5)
	    ax[1].plot(t,pred['Nmax']*norm.cdf(tau),label='prediction')
	    ax[1].plot([pred_date,pred_date],[data.min(),3*data.max()],'k--',label='prediction date')
	else:
	    ax[1].text(0.05,0.88,'No prediction available',fontsize=11,transform=ax[1].transAxes)
	ax[1].set_yscale('log')
	ax[1].set_ylabel('Cumulative confirmed cases')
	ax[1].set_ylim((thresh,None))
	ax[1].xaxis.set_major_locator(mdates.MonthLocator())
	ax[1].xaxis.set_minor_locator(mdates.DayLocator())
	ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
	plt.show()