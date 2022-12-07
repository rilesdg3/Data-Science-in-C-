/*
 * Stats.cpp
 *
 *  Created on: Mar 18, 2019
 *      Author: ryan
 */

#include "Data.h"
#include "Stats.h"


using namespace std;
namespace Stats{


template<typename T>
std::map<std::string, double > CalcSimpleStats(vector<T > &data, Histogram &hist_st, int n_bins, bool show_stats){

	//int c = data.size();//data.size();//cache size for histogramm.

	std::map<std::string, double > my_stats;



	std::vector<T> my_data(data.size());
	auto end = std::copy_if (data.begin(), data.end(), my_data.begin(), [](T i){return !(std::isnan(i));} );
	my_data.resize(std::distance(my_data.begin(),end));

	int c = my_data.size();

	std::vector<int> sorted_index(my_data.size());//holds the index of my_data as if it were sorted, i.e. sorted_index[0] =12375 would be
	//either the min or max or whatever my_data[12375] sorted_index[1] = 896, would be the second min or max or whatever

	//std::vector<int> bin(my_data.size());
	hist_st.bin_index.resize(my_data.size());// = bin;
	std::iota(sorted_index.begin(), sorted_index.end(),0);
	std::sort(sorted_index.begin(), sorted_index.end(), [&my_data](T i1, T i2){return my_data[i1] < my_data[i2];} );
	//w_acc wh_acc(boost::accumulators::tag::density::num_bins = 20, boost::accumulators::tag::density::cache_size = 10);
	stat_acc wh_acc(boost::accumulators::tag::density::num_bins = n_bins, boost::accumulators::tag::density::cache_size = 10);

	//FOR THE CORRECT MEDIAN WHEN THERE IS AN EVEN NUMBER OF DATA POINTS
	//bacc::accumulator_set<double,
	//bacc::stats<bacc::tag::median(bacc::with_density) > >
	//    acc_median( bacc::density_cache_size = 4, bacc::density_num_bins = 4 );
	std::map<std::string, std::function<double()> > stats {
		{ "count",   [&wh_acc] () { return bacc::count(wh_acc);  }},
		{ "min",   [&wh_acc] () { return bacc::min(wh_acc);  }},
		{ "mean",  [&wh_acc] () { return bacc::mean(wh_acc); }},
		{ "median", [&wh_acc] () { return bacc::median(wh_acc); }},
		{ "max",   [&wh_acc] () { return bacc::max(wh_acc);  }},
		{ "skewness",[&wh_acc] () { return bacc::skewness(wh_acc);  }},
		{ "range", [&wh_acc] () { return (bacc::max(wh_acc) - bacc::min(wh_acc)); }},
		{ "var",   [&wh_acc] () {
			int n = bacc::count(wh_acc);
			double f = (static_cast<double>(n) / (n - 1));
			return f * bacc::variance(wh_acc);
		}},
		{ "sd",    [&stats] () { return std::sqrt(stats["var"]()); }}
	};


	acc myAccumulator( boost::accumulators::tag::density::num_bins = n_bins, boost::accumulators::tag::density::cache_size = 10);

	//fill accumulator
	for (int j = 0; j < c; ++j)
	{
		//why did I have this myAccumulator((double)my_data[j]);
		wh_acc((double)my_data[j]);
	}

	histogram_type hist = boost::accumulators::density(wh_acc);

	double total = 0.0;
	std::vector<int>::iterator it = sorted_index.begin();
	//int bin_number = 0;
	for( uint i = 0; i < hist.size(); i++ )
	{
		//std::cout << "Bin lower bound: " << hist[i].first << ", Value: " << hist[i].second << std::endl;
		hist_st.bins.push_back(hist[i].first);
		hist_st.marginals.push_back(hist[i].second);
		for(; (it!=sorted_index.end() && my_data[*it] < hist[i].first); ++it){
			hist_st.bin_index[*it]=i;
			//cout<<"my_data[*it] < hist[i].first "<<(my_data[*it] < hist[i].first)<<endl;
			//cout<<"value "<<my_data[*it]<<" n_bin "<<hist_st.bin_index[*it]<<" *it "<<*it<< " Bin lower bound: " << hist[i].first<<endl;

		}

		total += hist[i].second;
	}

	auto p = std::cout.precision();
	std::cout.precision(5);
	for(auto key = stats.begin(); key != stats.end(); ++key){
		my_stats[key->first] = key->second();
		//if(show_stats){
		//	cout<<key->first<<" "<<key->second()<<" "<<std::endl;
		//}
	}
	//if(show_stats)
	//	cout<<endl;

	if(show_stats){
		std::map<std::string, std::function<double()> >::iterator key;
		for(key = stats.begin(); key != stats.end(); ++key){
			std::cout<<key->first<<" "<<key->second()<<std::endl;
		}

	}


	//stats = clear_stats;

	std::cout.precision(p);
	return my_stats;


}
template std::map<std::string, double > CalcSimpleStats<float>(vector<float > &, Histogram &, int, bool);
template std::map<std::string, double > CalcSimpleStats<double>(vector<double> &, Histogram &, int, bool);
template std::map<std::string, double > CalcSimpleStats<long double>(vector<long double> &, Histogram &, int, bool);

/*
 * Plots a histogram
 */
void PlotHist(Histogram &hist_st, string file_path, string file_name){

	std::vector<std::pair<float, float>> data;
	std::vector<float> values;

	//cvplot::Window::current("cvplot demo").offset({60, 100});

/*
	{
		auto name = file_name;//"simple";
		//cvplot::setWindowTitle(name, "histogram");
		//cvplot::moveWindow(name, 0, 0);
		//cvplot::resizeWindow(name, 1000, 1000);
		auto &figure = cvplot::figure(name);
		//figure.series("line")
		//    .setValue({1.f, 2.f, 3.f, 4.f, 5.f})
		//    .type(cvplot::DotLine)
		//    .color(cvplot::Blue);
		figure.series("Hist")
            		.setValue(hist_st.marginals)
					.type(cvplot::Histogram)
					.color(cvplot::Red);
		figure.show(false);
		figure.save(file_path,file_name);
		figure.clear();
	}
*/


	//cv::waitKey(0);
}

template <typename T>
void PlotLine(std::vector<T> &data, string name){

	//std::vector<std::pair<float, float>> data;
	std::vector<float> values(data.begin(), data.end());

	//cvplot::Window::current("cvplot demo").offset({60, 100});

/*
	{
		//auto name = "simple";
		//cvplot::setWindowTitle(name, "line");
		//cvplot::moveWindow(name, 0, 0);
		//cvplot::resizeWindow(name, 1000, 1000);
		auto &figure = cvplot::figure(name);
		figure.origin(false,false);
		//figure.series("line")
		//    .setValue({1.f, 2.f, 3.f, 4.f, 5.f})
		//    .type(cvplot::DotLine)
		//    .color(cvplot::Blue);
		figure.series("Line")
            		.setValue(values)
					.type(cvplot::Line)
					.color(cvplot::Red);
		figure.show(false);
		figure.save("/home/ryan/LANL-Earthquake-Prediction/",name);
		figure.clear();
	}

*/

	//cv::waitKey(0);
}
template void PlotLine<double >(std::vector<double> & ,string );
template void PlotLine<float >(std::vector<float> & ,string );

/*
 * @brief: Computes a histogram and the corresponding histogram bin that the value falls into
 *
 * @tparam: std::vector<T> data T type of element
 * @param: Histogram hist_st historgram structure the histogram data will be stored here
 * @param: int n_bins number of bins in histogram
 */
template< typename T >
void ComputeHistograms(std::vector<T> &data,Histogram &hist_st, int n_bins){

	//int n_nan = std::count_if(data.begin(),data.end(), MyIsNan);


	int my_n_bins=  n_bins;
	std::vector<T> my_data(data.size());
	auto end = std::copy_if (data.begin(), data.end(), my_data.begin(), [](T i){return !(std::isnan(i));} );
	my_data.resize(std::distance(my_data.begin(),end));

	int c = my_data.size();

	hist_st.count = c;

	std::vector<int> sorted_index(my_data.size());//holds the index of my_data as if it were sorted, i.e. sorted_index[0] =12375 would be
	//either the min or max or whatever my_data[12375] sorted_index[1] = 896, would be the second min or max or whatever

	//std::vector<int> bin(my_data.size());
	hist_st.bin_index.resize(my_data.size());// = bin;
	std::iota(sorted_index.begin(), sorted_index.end(),0);
	std::sort(sorted_index.begin(), sorted_index.end(), [&my_data](T i1, T i2){return my_data[i1] < my_data[i2];} );


	 //boost::math::statistics::interquartile_range(data);

	int err = 0;
	if(n_bins == 0){
		double high = my_data[sorted_index[sorted_index.size()-1]];
		double low = my_data[sorted_index[0]];
		std::vector<double> for_gretl(my_data.size());
		for(uint i = 0; i<my_data.size(); ++i)
			for_gretl[i] = my_data[i];

		double iqr = gretl_quantile(0, for_gretl.size()-1, for_gretl.data(), 0.75, OPT_NONE, &err);
		iqr -= gretl_quantile(0, for_gretl.size()-1, for_gretl.data(),0.25, OPT_NONE, &err);
		my_n_bins = (high - low)/(2*iqr*pow(c,(-1/3)) );
		if(my_n_bins <= 1)
			my_n_bins = 1;
	}

	stat_acc wh_acc(boost::accumulators::tag::density::num_bins = my_n_bins, boost::accumulators::tag::density::cache_size = c);

	//fill accumulator
	for (int j = 0; j < c; ++j)
	{
		//myAccumulator(my_data[j]);
		wh_acc(my_data[j]);
	}

	//histogram_type hist = boost::accumulators::density(myAccumulator);

	histogram_type hist = boost::accumulators::density(wh_acc);

	double total = 0.0;
	std::vector<int>::iterator it = sorted_index.begin();
	//int bin_number = 0;
	for( uint i = 0; i < hist.size(); i++ )
	{
		//std::cout << "Bin lower bound: " << hist[i].first << ", Value: " << hist[i].second << std::endl;
		hist_st.bins.push_back(hist[i].first);
		hist_st.marginals.push_back(hist[i].second);
		for(; (it!=sorted_index.end() && my_data[*it] < hist[i].first); ++it){
			hist_st.bin_index[*it]=i;
			//cout<<"my_data[*it] < hist[i].first "<<(my_data[*it] < hist[i].first)<<endl;
			//cout<<"value "<<my_data[*it]<<" n_bin "<<hist_st.bin_index[*it]<<" *it "<<*it<< " Bin lower bound: " << hist[i].first<<endl;

		}

		total += hist[i].second;
	}

	//std::cout << "Total: " << total << std::endl; //should be 1 (and it is)

}
template void ComputeHistograms<double >(std::vector<double> & ,Histogram &, int);

template<typename T>
void ComputeHistograms(DATASET *dset, Histogram &hist_st, int varno, int n_bins){

	std::vector<double> my_data(dset->n);
	int my_n_bins=  n_bins;

	auto end = std::copy_if(dset->Z[varno], dset->Z[varno]+dset->n, my_data.begin(), [](double i){return !(std::isnan(i));} );

	my_data.resize(std::distance(my_data.begin(),end));

	int c = my_data.size();

	hist_st.count = c;

	std::vector<int> sorted_index(my_data.size());//holds the index of my_data as if it were sorted, i.e. sorted_index[0] =12375 would be
	//either the min or max or whatever my_data[12375] sorted_index[1] = 896, would be the second min or max or whatever

	//std::vector<int> bin(my_data.size());
	hist_st.bin_index.resize(my_data.size());// = bin;
	std::iota(sorted_index.begin(), sorted_index.end(),0);
	std::sort(sorted_index.begin(), sorted_index.end(), [&my_data](double i1, double i2){return my_data[i1] < my_data[i2];} );

	int err = 0;
	if(n_bins == 0){
		double high = my_data[sorted_index[sorted_index.size()-1]];
		double low = my_data[sorted_index[0]];
		double iqr = gretl_quantile(0, my_data.size()-1, my_data.data(), 0.75, OPT_NONE, &err);

		iqr -= gretl_quantile(0, my_data.size()-1, my_data.data(), 0.25, OPT_NONE, &err);

		my_n_bins = (high - low)/(2*iqr*pow(c,(-1/3)) );
		if(my_n_bins <= 1)
			my_n_bins = 1;
	}

	stat_acc wh_acc(boost::accumulators::tag::density::num_bins = my_n_bins, boost::accumulators::tag::density::cache_size = c);

	//fill accumulator
	for (int j = 0; j < c; ++j)
	{
		//myAccumulator(my_data[j]);
		wh_acc(my_data[j]);
	}

	//histogram_type hist = boost::accumulators::density(myAccumulator);

	histogram_type hist = boost::accumulators::density(wh_acc);

	double total = 0.0;
	std::vector<int>::iterator it = sorted_index.begin();
	//int bin_number = 0;
	for( uint i = 0; i < hist.size(); i++ )
	{
		//std::cout << "Bin lower bound: " << hist[i].first << ", Value: " << hist[i].second << std::endl;
		hist_st.bins.push_back(hist[i].first);
		hist_st.marginals.push_back(hist[i].second);
		for(; (it!=sorted_index.end() && my_data[*it] < hist[i].first); ++it){
			hist_st.bin_index[*it]=i;
			//cout<<"my_data[*it] < hist[i].first "<<(my_data[*it] < hist[i].first)<<endl;
			//cout<<"value "<<my_data[*it]<<" n_bin "<<hist_st.bin_index[*it]<<" *it "<<*it<< " Bin lower bound: " << hist[i].first<<endl;

		}

		total += hist[i].second;
	}
}
template void ComputeHistograms<double >(DATASET * ,Histogram &, int, int);


/*
 *
 *
 * return vector<double>(4) [0]=first_min value, [1]=lag were first_min occured, [2]=min value, [3]=lag were min occured
 */
template<typename T, typename T1>
std::vector<double> LaggedMI(std::vector<T> pred_vars, std::vector<T1> target, int n_bins, int min_lag, int max_lag, int lag_step){


	//cout<<"LaggedMI start"<<endl;
	int first_min_index = 0;
	int min_index = 0;
	double first_min = std::numeric_limits<double>::max();
	double min = std::numeric_limits<double>::max();
	double mi = 0.0;
	bool found_first_min = false;
	int len = (max_lag-min_lag)/lag_step;
	std::vector<double > mi_values(len+1);
	int mi_values_it = 0;


	std::vector<std::pair<double, double>> xy_pts_A(mi_values.size()-2);

	std::vector<double> return_values(4);

	//std::vector<double > pred_vars_sub(pred_vars.begin(),pred_vars.end());
	//std::vector<double > target_sub(target.begin(),target.end());


	std::vector<T> my_pred_vars(pred_vars.size());
	std::vector<T> my_target(target.size());

	//missing data with time series funcks up the datay because if I drop a day it is now two days
	//but if I drop the day I am also saying that day did not occur


	//here fixing this so it handles nan values
	//also need to adjust the one below as well

	std::set<int> indexes_to_remove;

	int size_less_nan = target.size();
	int my_iter = 0;

	if(pred_vars.size()==target.size() ){

		for(int i=0; i<target.size();++i){
			if(!std::isnan(pred_vars[i]) && !std::isnan(target[i])){
				my_pred_vars[my_iter] = pred_vars[i];
				my_target[my_iter] = target[i];
				++my_iter;
			}
			else
				--size_less_nan;

		}

		assert(size_less_nan == my_iter);
	my_pred_vars.resize(std::distance(my_pred_vars.begin(), my_pred_vars.begin()+my_iter ) );
	my_target.resize(std::distance(my_target.begin(),my_target.begin()+my_iter ) );


		//cout<<"Entropy "<<Stats::Entropy(h1)<<endl;

		for(int i = min_lag; i<=max_lag; i=i+lag_step){
			std::vector<T > pred_vars_sub(my_pred_vars.begin(),my_pred_vars.end()-i);
			std::vector<T1> target_sub(my_target.begin()+i,my_target.end());

			Stats::Histogram h;
			Stats::Histogram h1;
			Stats::ComputeHistograms(target_sub, h1,n_bins);
			Stats::ComputeHistograms(pred_vars_sub, h,n_bins);
			mi=Stats::DiscreteMI(h,h1);
			//cout<<"MI "<<mi<<endl;

			if(i >1){
				if(found_first_min == false){
					if(first_min >= mi){
						first_min = mi;
						first_min_index = i;
					}
					else
						found_first_min = true;
				}
				if(min >= mi){
					min = mi;
					min_index = i;
				}
				//mi_values[i] = mi;
				mi_values[mi_values_it] = mi;
				if(i < max_lag)
					xy_pts_A[mi_values_it] = std::make_pair(i,mi);
				mi_values_it++;
			}


		}

		cout<<"First Min "<< first_min<<" Found at "<<first_min_index<<endl;
		cout<<"Min "<< min<<" Found at "<<min_index<<endl;
		return_values[0] = first_min;
		return_values[1] = first_min_index;
		return_values[2] = min;
		return_values[3] = min_index;
	}
	else{
		return_values[0] = std::nan("");
		return_values[1] =  std::nan("");
		return_values[2] =  std::nan("");
		return_values[3] =  std::nan("");

	}
	//cout<<"LaggedMI end"<<endl;
	//PlotLine(mi_values, "LaggedMI");


	Gnuplot gp;
		// For debugging or manual editing of commands:
		//Gnuplot gp(std::fopen("plot.gnu", "w"));
		// or
		//Gnuplot gp("tee plot.gnu | gnuplot -persist");

	//	gp<<"set xtics rotate\n";
		//gp << "set xrange ["+dates[0]+":"+dates[dates.size()-1]+"]\nset yrange ["+std::to_string(*min)+":"+std::to_string(*max)+"]\n";
	//	gp << "set xrange [0:"+std::to_string(f.size()-1)+"]\nset yrange ["+std::to_string(*fmin)+":"+std::to_string(*fmax)+"]\n";

	//	gp.precision(5);
	//	gp<<"set format \"%5.5f\"\n";
		//gp<<"set term png size 15000 15000\n";
		gp<<"set term png size 700,300\n";
		gp<<"set output "<<"'/home/ryan/Nifty/LaggedMI.png'"<<"\n";
		//gp<<"set xtics 0,.2,10\n";
		//gp<<"plot '-' using 2:xtic(1) with lines\n";
		//gp<<"plot '-' using 2:xtic(1) with histogram\n";
		//gp<<"plot '-' using 2 with lines\n";
		//gp<<"set style lines\n";
		gp<<"plot '-' using 2 with lines\n";
		gp.send1d(xy_pts_A);

	return return_values;

}
template std::vector<double> LaggedMI<float, double>(std::vector<float> pred_vars, std::vector<double> target, int n_bins, int min_lag, int max_lag, int lag_step);
template std::vector<double> LaggedMI<double, double>(std::vector<double> pred_vars, std::vector<double> target, int n_bins, int min_lag, int max_lag, int lag_step);


void LaggedMI(std::vector<std::vector<double> > embedVect, std::vector<double > vect_y_alligned, int n_bins){


	int first_min_index = 0;
	int min_index = 0;
	double first_min = std::numeric_limits<double>::max();
	double min = std::numeric_limits<double>::max();
	double mi = 0.0;
	bool found_first_min = false;
	std::vector<double > mi_values(embedVect.size());
	std::vector<double > tmp(embedVect.size());

	Stats::Histogram h1;
	Stats::ComputeHistograms(vect_y_alligned, h1,n_bins);
	cout<<"Entropy "<<Stats::Entropy(h1)<<endl;


	//for(uint i = 0; i<embedVect[0].size(); ++i){
	for(int i = embedVect[0].size(); i>=0; --i){
			Stats::Histogram h;
			for(uint j =0; j< embedVect.size(); ++j)
				{//cout<<"j "<<j<<endl;
				tmp[j] = embedVect[j][i];
				}
			Stats::ComputeHistograms(tmp, h,n_bins);
			mi=Stats::DiscreteMI(h,h1);
			//cout<<"MI "<<mi<<endl;

			if(found_first_min == false){
				if(first_min >= mi){
					first_min = mi;
					first_min_index = i;
				}
				else
					found_first_min = true;
			}
			if(min >= mi){
				min = mi;
				min_index = i;
			}

			mi_values[i] = mi;
			//tmp.clear();
		}

	//PlotLine(mi_values, "LaggedMI");
	cout<<"First Min "<< first_min<<" Found at "<<first_min_index<<endl;
	cout<<"Min "<< min<<" Found at "<<min_index<<endl;

}
double DiscreteMI(Histogram &pred_hist_st, Histogram &target_hist_st)
{

	int n_cases = pred_hist_st.bin_index.size()-1;//number of rows in vector
	int n_bins_pred = pred_hist_st.marginals.size();
	int n_bins_target = target_hist_st.marginals.size();
	std::vector<int > temp(target_hist_st.bins.size());
	std::vector<std::vector<int > > contingency_table(target_hist_st.bins.size(),temp);


	pred_hist_st.contingency_table.resize(pred_hist_st.bins.size());
	for(int i =0; i<pred_hist_st.contingency_table.size();++i)
		pred_hist_st.contingency_table[i].resize(target_hist_st.bins.size());

	int i, j ;
	double px, py, pxy, MI ;

	/*for (i=0 ; i<n_bins_pred ; i++) {      // Zero bin counts
      for (j=0 ; j<n_bins_target ; j++)
         bin_counts[i*n_bins_target+j] = 0 ;
      }*/


	for(uint it = 0; it<pred_hist_st.bin_index.size(); ++it){
		pred_hist_st.contingency_table[pred_hist_st.bin_index[it]][target_hist_st.bin_index[it]]++;
		//for(int target_it = 0; target_it<target_hist_st.bin_index.size(); ++target_it)
	}

	//for (i=0 ; i<ncases ; i++)
	//   ++bin_counts[pred_bin[i]*nbins_target+target_bin[i]] ;

	MI = 0.0 ;
	for (i=0 ; i<n_bins_pred ; i++) {
		px = pred_hist_st.marginals[i]; //pred_marginal[i] ;
		for (j=0 ; j<n_bins_target ; j++) {
			py = target_hist_st.marginals[j];//target_marginal[j] ;
			pxy = (double) pred_hist_st.contingency_table[i][j]/ (double) n_cases;//(double) bin_counts[i*n_bins_target+j] / (double) n_cases ;
			if (pxy > 0.0 && px > 0.0 && py > 0.0)
				MI += pxy * log ( pxy / (px * py) ) ;
		}
	}

	return MI ;
}

/*
 * @brief: Computes Entropy
 *
 * @param: Histogram hist_st historgram structure that is used to calculate the entropy
 *
 * @return: double the entropy
 */
double Entropy (Histogram &hist_st)
{
   double p, ent ;

   ent = 0.0 ;
   for( auto e : hist_st.marginals){
	   p = e;
	   if(p!=0)
		   ent += p*log(p);

   }
   return -ent ;
}

/*
 *
 */
void ACF(std::vector<std::vector<double> > &embedVect, int lag){

	std::vector<double> corr_vect;
	double corr = 0.0;

	for(int i = embedVect.size()-1; i>=lag; --i){
		corr = correlation(i-lag, embedVect,embedVect[embedVect.size()-1]);
		corr_vect.push_back(corr);
		cout<<"ACF: lag "<<embedVect.size()-i<<" corr "<<corr<<endl;
	}
	PlotLine(corr_vect,"ACF");


}


/*
 * @brief: computes pearson correlation R
 *
 * @param int varnum column number of variable
 * @tparam_T vector<vector<T> > data: martix of variables
 * @tparam_T1 vector<T1> target: target variable
 */
template<typename T, typename T1>
double correlation(int var_num, std::vector<std::vector<T> > &data, std::vector<T1> &target)
{

	//cout<<typeid(data).name()<<endl;// returns something like this St6vectorIS_IdSaIdEESaIS1_EE and you can then search the string to find the type
	int n_cases = data[var_num].size()-1;//number of rows in vector
	int icase ;
	double xdiff, ydiff, xmean, ymean, xvar, yvar, xy ;

	xmean = ymean = 0.0 ;
	for (icase=0 ; icase<n_cases ; icase++) {
		xmean += data[var_num][icase];
		ymean += target[icase] ;
	}
	xmean /= n_cases ;
	ymean /= n_cases ;

	xvar = yvar = xy = 1.e-30 ;
	for (icase=0 ; icase<n_cases ; icase++) {
		xdiff = data[var_num][icase];
		ydiff = target[icase] - ymean ;
		xvar += xdiff * xdiff ;
		yvar += ydiff * ydiff ;
		xy += xdiff * ydiff ;
	}

	return xy / sqrt ( xvar * yvar ) ;
}
template double correlation<double, double>(int, std::vector<std::vector<double> > &, std::vector<double> &);

/*
 * @brief: computes pearson correlation R
 *
 * @param int varnum column number of variable
 * @tparam_T vector<vector<T> > data: martix of variables
 * @tparam_T1 vector<T1> target: target variable
 */
template<typename T, typename T1>
double correlation(std::vector<T> &data, std::vector<T1> &target)
{

	//cout<<typeid(data).name()<<endl;// returns something like this St6vectorIS_IdSaIdEESaIS1_EE and you can then search the string to find the type
	int n_cases = data.size();//number of rows in vector

	int icase ;
	double xdiff, ydiff, xmean, ymean, xvar, yvar, xy ;


	xmean = ymean = 0.0 ;
	for (icase=0 ; icase<data.size() ; icase++) {
		if( !std::isnan(data[icase]) && !std::isnan(target[icase]) ){
			xmean += data[icase];
			ymean += target[icase] ;
		}
		else
			--n_cases;
	}
	xmean /= n_cases ;
	ymean /= n_cases ;

	xvar = yvar = xy = 1.e-30 ;
	for (icase=0 ; icase<data.size() ; icase++) {
		if(!std::isnan(data[icase]) && !std::isnan(target[icase]) ){
			xdiff = data[icase];
			ydiff = target[icase] - ymean ;
			xvar += xdiff * xdiff ;
			yvar += ydiff * ydiff ;
			xy += xdiff * ydiff ;
		}
	}

	return xy / sqrt ( xvar * yvar ) ;
}
template double correlation<double, double>(std::vector<double> &, std::vector<double> &);

/*
 * Rounds a number to the nearest value
 *
 * ex: number = 1.37, value = .25 return = 1.50, is value = .25 then it will round number up/down to nearest .25
 */
template<typename T, typename T1>
T RoundToNearestValue(T number, T1 value){

	int quotient;
	return number-remquo(number,value,&quotient);
}
template double  RoundToNearestValue<double, double>(double, double);

/*
 * THIS WAS NOT WORKING FOR THE LANL DATA SET
 * I think the reason is there are to many bins with a zero count
 * https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Problems
 *
 */
double ComputeV (Histogram &pred_hist_st, Histogram &target_hist_st)
{

	int n_cases = pred_hist_st.bin_index.size()-1;//number of rows in vector
	int n_bins_pred = pred_hist_st.marginals.size()-1;
	int n_bins_target = target_hist_st.marginals.size()-1;
	std::vector<int > temp(target_hist_st.bins.size());
	std::vector<std::vector<int > > contingency_table(target_hist_st.bins.size(),temp);


	pred_hist_st.contingency_table.resize(pred_hist_st.bins.size());
	for(int i =0; i<pred_hist_st.contingency_table.size();++i)
		pred_hist_st.contingency_table[i].resize(target_hist_st.bins.size());


	int i, j ;
	double diff, expected, chisq, V = 0.0;

	/*for (i=0 ; i<nbins_pred ; i++) {      // Zero bin counts
		for (j=0 ; j<nbins_target ; j++)
			bin_counts[i*nbins_target+j] = 0 ;
	}*/

	for(uint it = 0; it<pred_hist_st.bin_index.size(); ++it){
		pred_hist_st.contingency_table[pred_hist_st.bin_index[it]][target_hist_st.bin_index[it]]++;
		//for(int target_it = 0; target_it<target_hist_st.bin_index.size(); ++target_it)
	}
	//for (i=0 ; i<n_cases ; i++)
	//	++bin_counts[pred_bin[i]*nbins_target+target_bin[i]] ;

	//so bin_counts = the number of values that fall into bin_i
	// so I think this would be N_x(i) =  number of x's that fall into bin i-> in short histogram values
	// F_x(i)= marginal of x in bin i


	chisq = 0.0 ;
	//F_x(i)= marginal of x in bin i
	//Calc F_x,y(i,j)=F_x(i)F_y(j)
	//Calc E_i,j)=NF_x,y(i,j) Expected
	for (i=0 ; i<n_bins_pred ; i++) {
		for (j=0 ; j<n_bins_target ; j++) {
			expected = pred_hist_st.marginals[i]*target_hist_st.marginals[j]*n_cases;//pred_marginal[i] * target_marginal[j] * n_cases ;
			diff = pred_hist_st.contingency_table[i][j]-expected;//bin_counts[i*nbins_target+j] - expected ;
			cout<<"expected "<<expected<<" diff "<<diff<<" (double)expected + .000000000001 "<<((double)expected + .000000000001)<<endl;
			if(expected>0){
				chisq += diff * diff / (expected) ;
			}
			cout<<"chisq "<<chisq<<endl;
		}
	}

	V = chisq / n_cases ;
	if (n_bins_pred < n_bins_target)
		V /= n_bins_pred - 1 ;
	else
		V /= n_bins_target - 1 ;

	V = sqrt ( V ) ;

	return V ;
}

vector<vector<int> > Combinations(vector<int> iterable, int r){
	//# combinations('ABCD', 2) --> AB AC AD BC BD CD
	//# combinations(range(4), 3) --> 012 013 023 123
	//CreateSets();

	int max = *std::max(iterable.begin(),iterable.begin()+iterable.size()-1);
	//auto pool = iterable;
	int n = iterable.size();//len(pool)
	vector<int >ranges;
	vector<vector<int> > all_combinations;
	vector<int > tmp(r);

	if (r > n)
		return all_combinations;
	//indices = list(range(r))
	vector<int> indices;
	for(int it = 0; it<r; it++)
		indices.push_back(it);

	//yield tuple(pool[i] for i in indices)
	for(auto it = 0; it < indices.size(); ++it)
		tmp[it] = indices[it];//pool[it] = indices[it];
	if(*std::max(tmp.begin(),tmp.begin()+tmp.size()-1)<=max)
		all_combinations.push_back(tmp);

	for(int i =0; i<r;++i)
		ranges.push_back(i);

	std::reverse(ranges.begin(),ranges.end());
	int i= ranges[0];
	while(indices[r-1]<n){
		for(; i>ranges[ranges.size()-1]; --i){//in reversed(range(r)):
			if(indices[i] != i + n - r)
				break;
		}
		indices[i] += 1;

		for(int j = i+1; j<r;++j)
			indices[j] = indices[j-1] + 1;

		for(auto it = 0; it < indices.size(); ++it)
			tmp[it] = indices[it];//pool[it] = indices[it];
		//cout<<"tmp "<<tmp<<endl;
		//cout<<"indices "<<indices<<endl;
		if(*std::max(tmp.begin(),tmp.begin()+tmp.size()-1)<=max)
			all_combinations.push_back(tmp);
		i= ranges[0];
	}
	//cout<<"tmp "<<tmp<<endl;
	//cout<<"all_combinations "<<all_combinations[0]<<endl;
	return all_combinations;
}

template<typename T>
double Mean(std::vector<T> data, int start, int end){


	std::vector<T> my_vector;
	typename decltype(my_vector)::iterator myvectit;
	std::vector<double>::iterator my_vector_iter;
	//const std::type_info  &ti = typeid(tmp_vect);
	//typedef typename T my_type;
	//decltype(T) my_type;
	//std::vector<decltype(T)>::iterator iter_begin = data.begin();
	//std::vector<>::iterator it = v.begin();
	//std::advance( it, index );
	if(start != -1 && start != end){


		//std::vector<my_type>::iterator iter_begin = data.begin();
		//std::vector<double>::iterator iter_end;
		//std::advance(iter_begin, start );
		//std::advance( iter_end, iter_end);

	}

	return  boost::math::statistics::mean(data.begin(), data.end());
}
template double Mean<double>(std::vector<double>,int ,int );

template<typename T>
double StdDev(std::vector<T> data){

	return std::sqrt(boost::math::statistics::variance(data.begin(), data.end()));
}

template<typename T>
std::vector<double> DiffSeries(std::vector<T> &data){

	std::vector<double> return_data(data.size());

	int old_value=0;
	return_data[0] = std::nan("");
	for(uint new_value=1; new_value<data.size(); ++new_value,++old_value ){
		return_data[new_value] = ((double)data[new_value]-(double)data[old_value]);

	}

	return return_data;

}
template std::vector<double> DiffSeries<double>(std::vector<double> &);
template std::vector<double> DiffSeries<long double>(std::vector<long double> &);

void two_samples_t_test_equal_sd(
        double Sm1,
        double Sd1,
        unsigned Sn1,
        double Sm2,
        double Sd2,
        unsigned Sn2,
        double alpha)
{
   //
   // Sm1 = Sample Mean 1.
   // Sd1 = Sample Standard Deviation 1.
   // Sn1 = Sample Size 1.
   // Sm2 = Sample Mean 2.
   // Sd2 = Sample Standard Deviation 2.
   // Sn2 = Sample Size 2.
   // alpha = Significance Level.
   //
   // A Students t test applied to two sets of data.
   // We are testing the null hypothesis that the two
   // samples have the same mean and that any difference
   // if due to chance.
   // See http://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm
   //
   using namespace std;
   using namespace boost::math;

   // Print header:
   cout <<
      "_______________________________________________\n"
      "Student t test for two samples (equal variances)\n"
      "_______________________________________________\n\n";
   cout << setprecision(5);
   cout << setw(55) << left << "Number of Observations (Sample 1)" << "=  " << Sn1 << "\n";
   cout << setw(55) << left << "Sample 1 Mean" << "=  " << Sm1 << "\n";
   cout << setw(55) << left << "Sample 1 Standard Deviation" << "=  " << Sd1 << "\n";
   cout << setw(55) << left << "Number of Observations (Sample 2)" << "=  " << Sn2 << "\n";
   cout << setw(55) << left << "Sample 2 Mean" << "=  " << Sm2 << "\n";
   cout << setw(55) << left << "Sample 2 Standard Deviation" << "=  " << Sd2 << "\n";
   //
   // Now we can calculate and output some stats:
   //
   // Degrees of freedom:
   double v = Sn1 + Sn2 - 2;
   cout << setw(55) << left << "Degrees of Freedom" << "=  " << v << "\n";
   // Pooled variance:
   double sp = sqrt(((Sn1-1) * Sd1 * Sd1 + (Sn2-1) * Sd2 * Sd2) / v);
   cout << setw(55) << left << "Pooled Standard Deviation" << "=  " << v << "\n";
   // t-statistic:
   double t_stat = (Sm1 - Sm2) / (sp * sqrt(1.0 / Sn1 + 1.0 / Sn2));
   cout << setw(55) << left << "T Statistic" << "=  " << t_stat << "\n";
   //
   // Define our distribution, and get the probability:
   //
   students_t dist(v);
   double q = cdf(complement(dist, fabs(t_stat)));
   cout << setw(55) << left << "Probability that difference is due to chance" << "=  "
      << setprecision(3) << scientific << 2 * q << "\n\n";
   //
   // Finally print out results of alternative hypothesis:
   //
   cout << setw(55) << left <<
      "Results for Alternative Hypothesis and alpha" << "=  "
      << setprecision(4) << fixed << alpha << "\n\n";
   cout << "Alternative Hypothesis              Conclusion\n";
   cout << "Sample 1 Mean != Sample 2 Mean       " ;
   if(q < alpha / 2)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Sample 1 Mean <  Sample 2 Mean       ";
   if(cdf(dist, t_stat) < alpha)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Sample 1 Mean >  Sample 2 Mean       ";
   if(cdf(complement(dist, t_stat)) < alpha)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << endl << endl;
}

void two_samples_t_test_unequal_sd(
        double Sm1,
        double Sd1,
        unsigned Sn1,
        double Sm2,
        double Sd2,
        unsigned Sn2,
        double alpha)
{
   //
   // Sm1 = Sample Mean 1.
   // Sd1 = Sample Standard Deviation 1.
   // Sn1 = Sample Size 1.
   // Sm2 = Sample Mean 2.
   // Sd2 = Sample Standard Deviation 2.
   // Sn2 = Sample Size 2.
   // alpha = Significance Level.
   //
   // A Students t test applied to two sets of data.
   // We are testing the null hypothesis that the two
   // samples have the same mean and that any difference
   // if due to chance.
   // See http://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm
   //
   using namespace std;
   using namespace boost::math;

   // Print header:
   cout <<
      "_________________________________________________\n"
      "Student t test for two samples (unequal variances)\n"
      "_________________________________________________\n\n";
   cout << setprecision(5);
   cout << setw(55) << left << "Number of Observations (Sample 1)" << "=  " << Sn1 << "\n";
   cout << setw(55) << left << "Sample 1 Mean" << "=  " << Sm1 << "\n";
   cout << setw(55) << left << "Sample 1 Standard Deviation" << "=  " << Sd1 << "\n";
   cout << setw(55) << left << "Number of Observations (Sample 2)" << "=  " << Sn2 << "\n";
   cout << setw(55) << left << "Sample 2 Mean" << "=  " << Sm2 << "\n";
   cout << setw(55) << left << "Sample 2 Standard Deviation" << "=  " << Sd2 << "\n";
   //
   // Now we can calculate and output some stats:
   //
   // Degrees of freedom:
   double v = Sd1 * Sd1 / Sn1 + Sd2 * Sd2 / Sn2;
   v *= v;
   double t1 = Sd1 * Sd1 / Sn1;
   t1 *= t1;
   t1 /=  (Sn1 - 1);
   double t2 = Sd2 * Sd2 / Sn2;
   t2 *= t2;
   t2 /= (Sn2 - 1);
   v /= (t1 + t2);
   cout << setw(55) << left << "Degrees of Freedom" << "=  " << v << "\n";
   // t-statistic:
   double t_stat = (Sm1 - Sm2) / sqrt(Sd1 * Sd1 / Sn1 + Sd2 * Sd2 / Sn2);
   cout << setw(55) << left << "T Statistic" << "=  " << t_stat << "\n";
   //
   // Define our distribution, and get the probability:
   //
   students_t dist(v);
   double q = cdf(complement(dist, fabs(t_stat)));
   cout << setw(55) << left << "Probability that difference is due to chance" << "=  "
      << setprecision(3) << scientific << 2 * q << "\n\n";
   //
   // Finally print out results of alternative hypothesis:
   //
   cout << setw(55) << left <<
      "Results for Alternative Hypothesis and alpha" << "=  "
      << setprecision(4) << fixed << alpha << "\n\n";
   cout << "Alternative Hypothesis              Conclusion\n";
   cout << "Sample 1 Mean != Sample 2 Mean       " ;
   if(q < alpha / 2)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Sample 1 Mean <  Sample 2 Mean       ";
   if(cdf(dist, t_stat) < alpha)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Sample 1 Mean >  Sample 2 Mean       ";
   if(cdf(complement(dist, t_stat)) < alpha)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << endl << endl;
}

/*
 * @brief: iteate through an array looking for nan values.
 * @note: since I use nan to mark a missing value I need to search vector<string> as well
 *
 */
template<typename T>
void CountNANs(std::vector<T> &data, std::vector<int> &nan_location){


	if(typeid(T) != typeid(std::string)){
		for(uint i = 0; i<data.size()-1; ++i){
			if(data[i] != data[i])
				nan_location.push_back(i);

		}
	}

}
template void CountNANs<double>(std::vector<double> &, std::vector<int> &);
template void CountNANs<long double>(std::vector<long double> &, std::vector<int> &);

template<>
void CountNANs(std::vector<std::string> &data, std::vector<int> &nan_location){

		std::string my_nan = "nan";
		for(uint i = 0; i<data.size(); ++i){
			if(data[i].compare(my_nan) == 0)
				nan_location.push_back(i);

		}

}

/*
 * @brief: iter through an array looking for nan values.
 * @note: since I use nan to mark a missing value I need to search vector<string> as well
 *
 */
template<typename T>
void CountNANs(std::vector<std::vector<T> > &data, std::vector<int> &nan_location){


	/*if(typeid(T) != typeid(std::string)){
		for(int i = 0; i<data.size(); ++i){
			if(data[i] != data[i])
				nan_location.push_back(i);

		}
	}
	 */

}
template void CountNANs<double>(std::vector<std::vector<double> >&, std::vector<int> &);
template void CountNANs<long double>(std::vector<std::vector<long double> >&, std::vector<int> &);

template<>
void CountNANs(std::vector<std::vector<std::string> > &data, std::vector<int> &nan_location){


	/*if(typeid(T) == typeid(std::string)){

		for(int i = 0; i<data.size(); ++i){
			if(data[i] == "nan")
				nan_location.push_back(i);

		}
	}*/

}

template<typename T>
bool myfunction (T i, T j) {
	return (i==j);
}


/*
 * TODO add plot, check if should std::map<string,...> should be std::map<T,...>name_count
 *
 * @brief: 	Get's UniqueCounts for data in vector<T>, stores them in value_count if a value is alread stored in value_count
 * 			then the new value is added to the previous count
 *
 * @tparam 	data: 			the data
 * @param	value_count: 	holds the unique value and counts the # of occurrences
 * @param	skip_nan:		true if you do not want to count nan-> default = true
 * @param	cumulative:		default = ??? true to display the results of value_count which may contain counts from previous calls to this function
 *
 */
template<typename T>
void GetUniqueCounts(std::vector<T> &data, std::map<T, int> &value_count,bool skip_nan, bool cumulative){

	std::map<T, int> my_value_count;
	typename std::map<T, int>::iterator my_value_count_iter;
	int my_count = 0;
	std::vector<T> myvector (data);
	if(skip_nan){
		MyData::RemoveNANs(myvector);
		myvector.shrink_to_fit();
	}
	std::vector<T> unique_copy(myvector.size());



	std::sort (myvector.begin(),myvector.end());

	typename std::vector<T>::iterator it;
	it=std::unique_copy (myvector.begin(),myvector.end(),unique_copy.begin());

	unique_copy.resize(std::distance(unique_copy.begin(),it));
	unique_copy.shrink_to_fit();
	std::cout<<"N_Unique Values: "<<unique_copy.size()<<std::endl;
	std::cout <<"Name:	";
	for(typename std::vector<T>::iterator iter=unique_copy.begin(); iter!=unique_copy.end(); ++iter){
		my_count = std::count(myvector.begin(), myvector.end(), *iter);

		if(value_count.find(*iter) != value_count.end()){
			my_count += value_count[*iter];
			my_value_count[*iter] = my_count;//need to put in count
			value_count[*iter] = my_count;//need to put in count
		}
		else{
			my_value_count[*iter] = my_count;//need to put in count
			value_count[*iter] = my_count;
		}

		std::cout<<*iter<<",";
	}
	std::cout<<std::endl;
	//std::cout<<"i "<<i<<std::endl;
	int total=0;
	std::cout<<"Count:	";
	for(auto it:my_value_count){
		std::cout<<it.second<<",";
		total += it.second;
	}
	std::cout<<std::endl;
	std::cout<<"Percent: ";
	auto p = std::cout.precision();
	std::cout.precision(3);
	for(auto it:my_value_count){
		std::cout<<(double)it.second/(double)total<<",";
	}
	std::cout<<std::endl;
	std::cout.precision(p);
	std::cout<<"Total:	"<<total<<std::endl;;
	std::cout<<std::endl;



}
template void GetUniqueCounts<double>(std::vector<double> &, std::map<double, int> &, bool , bool);
template void GetUniqueCounts<long double>(std::vector<long double> &, std::map<long double, int> &, bool , bool);

template<>
void GetUniqueCounts(std::vector<std::string> &data, std::map<std::string, int> &value_count, bool skip_nan, bool cumulative){

	std::map<std::string, int> my_value_count;
	typename std::map<std::string, int>::iterator my_value_count_iter;
	int my_count = 0;
	std::vector<std::string> myvector (data);
	if(skip_nan){
		MyData::RemoveNANs(myvector);
		myvector.shrink_to_fit();
	}
	std::vector<std::string> unique_copy(myvector.size());



	std::sort (myvector.begin(),myvector.end());

	typename std::vector<std::string>::iterator it;
	it=std::unique_copy (myvector.begin(),myvector.end(),unique_copy.begin());

	unique_copy.resize(std::distance(unique_copy.begin(),it));
	std::cout<<"N_Unique Values: "<<unique_copy.size()<<std::endl;;
	std::cout <<"Name:	";
	for(typename std::vector<std::string>::iterator iter=unique_copy.begin(); iter!=unique_copy.end(); ++iter){
		my_count = std::count(myvector.begin(), myvector.end(), *iter);
		if(value_count.find(*iter) != value_count.end()){
			my_count += value_count[*iter];
			my_value_count[*iter] = my_count;//need to put in count
			value_count[*iter] = my_count;//need to put in count
		}
		else{
			my_value_count[*iter] = my_count;//need to put in count
			value_count[*iter] = my_count;
		}

		std::cout<<*iter<<",";
	}
	std::cout<<std::endl;

	int total=0;
	std::cout<<"Count:	";
	for(auto it:my_value_count){
		std::cout<<it.second<<",";
		total += it.second;
	}
	std::cout<<std::endl;
	std::cout<<"Percent: ";
	auto p = std::cout.precision();
	for(auto it:my_value_count){
		std::cout<<std::setprecision(3)<<(double)it.second/(double)total<<",";
	}
	std::cout.precision(p);
	std::cout<<std::endl;
	std::cout<<"Total:	"<<total<<std::endl;;
	std::cout<<std::endl;
}
template void GetUniqueCounts<std::string>(std::vector<std::string> &, std::map<std::string, int> &, bool, bool);


/*
 * @brief: iterate through vector<vector<T> > and call GetUniqueCounts
 *
 */
template<typename T>
void GetUniqueCounts(std::vector<std::vector<T> > &data, std::initializer_list<int> &columns_to_get, std::map<T, int> &value_count, bool skip_nan, bool cumulative){

	for(auto it:columns_to_get){
		GetUniqueCounts(data[it],value_count, skip_nan, cumulative);

	}

}
template void GetUniqueCounts<double>(std::vector<std::vector<double> > &, std::initializer_list<int> &, std::map<double, int> &, bool, bool);
template void GetUniqueCounts<long double>(std::vector<std::vector<long double> > &, std::initializer_list<int> &, std::map<long double, int> &, bool, bool);


/*
 * @brief: 	uses the scipy implementation of the lomb-scargle, but it converts frequencies into angular frequencies, and give the option to center values and
 * 			normalize the periodogram, autopower is based off of astropy autopower method
 * @tparam:	vector<T> 	&times:			times corresponding to data in values
 * @tparam:	vector<T1>	&values:			measurement values
 * @tparam:	vector<T2>	&frequencies:	frequencies for output of periodogram
 * @param:	bool 		center_data:	default = true: subtract the mean
 * @param:	string 		normalization:	default = standard: standardize the periodogram
 *
 */

template<typename T, typename T1, typename T2>
std::vector<double> LombScargle(std::vector<T> &times, const std::vector<T1> &values, std::vector<T2> &frequencies, bool center_data , std::string normalization){


	if(times.size()!= values.size())
		std::cerr<<"LombScargle times.size() != values.size()"<<std::endl;


	std::vector<double> periodogram(frequencies.size());

	double c_ = 0.0000;
	double s_ = 0.0000;

	double xc = 0.0000;//times_cos
	double xs = 0.0000;//times_sin
	double cc = 0.0000;
	double ss = 0.0000;
	double cs = 0.0000;
	double tau = 0.0000;

	double c_tau = 0.000;
	double s_tau = 0.000;
	double c_tau2 = 0.000;
	double s_tau2 = 0.000;
	double cs_tau = 0.000;

	double tmp = 0.000;
	double tmp1 = 0.000;
	double tmp2 = 0.000;
	double tmp3 = 0.000;


	double yy = 0.0000;
	int normalize = 0;

	std::vector<T1> my_values = values;

	if(center_data){
		double mean = boost::math::statistics::mean(my_values.begin(),my_values.end());
		for_each(my_values.begin(),my_values.end(), [mean](double &i){ i=i-mean;}   );
	}

	if(normalization.compare("standard") == 0){
		for_each(my_values.begin(),my_values.end(), [&yy](T1 &i){ yy+=std::pow(i,2);});
		yy/=my_values.size();
		normalize = 1;

	}
	//double var = boost::math::statistics::variance(my_values.begin(),my_values.end());

	//agular frequinces
	double pi=3.1415926535897932384626433832795029l;
	for_each(frequencies.begin(),frequencies.end(),[pi](double &i){i=i*pi*2;});

	for(uint i = 0; i< frequencies.size(); ++i){

		xc = 0.0000;
		xs = 0.0000;
		cc = 0.0000;
		ss = 0.0000;
		cs = 0.0000;

		for(uint k = 0; k<times.size(); ++k){
			c_ = std::cos(frequencies[i]*times[k]);
			s_ = std::sin(frequencies[i]*times[k]);
			xc += my_values[k]*c_;
			xs += my_values[k]*s_;
			cc += c_*c_;
			ss += s_*s_;
			cs += c_*s_;
		}

		//std::cout<<c_<<" "<<s_<<" "<<xc<<" "<<xs<<" "<<cc<<" "<<ss<<" "<<cs<<std::endl;

		tau = ::atan2(2*cs,cc-ss)/(2*frequencies[i]);

		c_tau = std::cos(frequencies[i] * tau);
		s_tau = std::sin(frequencies[i] * tau);
		c_tau2 = c_tau * c_tau;
		s_tau2 = s_tau * s_tau;
		cs_tau = 2 * c_tau * s_tau;
		//std::cout<<c_tau<<" "<<s_tau<<" "<<c_tau2<<" "<<s_tau2<<" "<<cs_tau<<std::endl;

		tmp = (c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs);
		tmp1 = (c_tau2 * cc + cs_tau * cs + s_tau2 * ss);

		tmp2 = (c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc);
		tmp3 = (c_tau2 * ss - cs_tau * cs + s_tau2 * cc);

		//std::cout<<tmp<<" "<<tmp3<<std::endl;

		periodogram[i] = 0.5 * ((tmp /tmp1) + (tmp2 /tmp3));
		if(normalize ==1)
			periodogram[i] *= 2/(times.size()* yy);

	}

	//remove angular frequencies
	for_each(frequencies.begin(),frequencies.end(),[pi](double &i){i/=pi*2;});

	return periodogram;


}
template std::vector<double> LombScargle<double,double,double>(std::vector<double> &, const std::vector<double> &, std::vector<double> &, bool , std::string);
template std::vector<double> LombScargle<int,double,double>(std::vector<int> &, const std::vector<double> &, std::vector<double> &, bool , std::string);


/*
 *
 * @brief: Determine a suitable frequency grid for data. This is a C++ implmentation of https://github.com/astropy/astropy
 *
 * Note: that this assumes the peak width is driven by the observational baseline, which is generally a good assumption when the baseline is
 * 		much larger than the oscillation period. If you are searching for periods longer than the baseline of your observations, this may
 * 		not perform well.
 *
 * 		Even with a large baseline, be aware that the maximum frequency returned is based on the concept of "average Nyquist frequency", which
 * 		may not be useful for irregularly-sampled data. The maximum frequency can be adjusted via the nyquist_factor argument, or through the
 * 		maximum_frequency argument.
 *
 * @param optional<double>	samples_per_peak:	defualt = 5: The approximate number of desired samples across the typical peak
 * @param optional<double>	nyquist_factor:		defualt = 5: The multiple of the average nyquist frequency used to choose the maximum frequency if maximum_frequency is not provided.
 * @param optional<double>	minimum_frequency:	If specified, then use this minimum frequency rather than one chosen based on the size of the baseline.
 * @param optional<double>	maximum_frequency:	If specified, then use this maximum frequency rather than one chosen based on the average nyquist frequency.
 * @param optioanl<bool>	return_freq_limits: default = false: if true, return only the frequency limits rather than the full frequency grid.
 *
 * @treturn					frequency: 			return_freq_limits == false then The heuristically-determined optimal frequency bin else
 * 												return[0] = minimum_frequency, and return[1] = maximum_frequency
 *
 */
template<typename T, typename T1>
std::vector<T> AutoFrequency(std::vector<T1> &times, std::experimental::optional<double> samples_per_peak, std::experimental::optional<double> nyquist_factor, std::experimental::optional<double> minimum_frequency,
		std::experimental::optional<double> maximum_frequency,std::experimental::optional<bool> return_freq_limits){



	//std::experimental::optional op;

	samples_per_peak = samples_per_peak.value_or(5.0);

	nyquist_factor= nyquist_factor.value_or(5.0);
	minimum_frequency=minimum_frequency.value_or(0.0);
	maximum_frequency=maximum_frequency.value_or(0.00);
	return_freq_limits=return_freq_limits.value_or(false);

	std::cout<<"samples_per_peak "<< *samples_per_peak<<" nyquist_factor "<<*nyquist_factor<<" minimum_frequency "<<*minimum_frequency<< " maximum_frequency "<< *maximum_frequency<<" return_freq_limits "<< *return_freq_limits<<std::endl;

	auto [fmin, fmax] = std::minmax_element(std::begin(times), std::end(times));


	T1 baseline = *fmax - *fmin;
	int n_samples = times.size();

	double df = 1.0 / baseline / *samples_per_peak;

	if(*minimum_frequency == 0 )
		*minimum_frequency = 0.5 * df;

	if(*maximum_frequency == 0){
		double avg_nyquist = 0.5 * n_samples / baseline;
		maximum_frequency = (*nyquist_factor) * avg_nyquist;

	}


	int Nf = 1 + ::round(((*maximum_frequency - *minimum_frequency) / df));

	if( *return_freq_limits){
		std::vector<T> frequencies(2);
		frequencies[0] = *minimum_frequency;
		frequencies[1] = *minimum_frequency + df * (Nf - 1);
		return frequencies;
	}

	else{
		int count = 0;
		std::vector<T> frequencies(Nf,0.00);
		//for_each(frequencies.begin(),frequencies.end(), [&count](double &i){ i=count; ++count;});

		for_each(frequencies.begin(),frequencies.end(), [&df, &Nf, &minimum_frequency, &count](T &i){ i = *minimum_frequency + df * count; ++count;});
		return frequencies;
	}




}
template std::vector<double> AutoFrequency<double, double>(std::vector<double> &times, std::experimental::optional<double> samples_per_peak, std::experimental::optional<double> nyquist_factor, std::experimental::optional<double> minimum_frequency,
		std::experimental::optional<double> maximum_frequency,std::experimental::optional<bool> return_freq_limits);









/*
 *
 * @brief 	Level of maximum at a given false alarm probability. This gives an estimate of the periodogram level corresponding to a
 * 			specified false alarm probability for the largest peak, assuming a
 * 			null hypothesis of non-varying data with Gaussian noise.
 *
 * 			 The data needs to be centered-> this is done in LombScargle
 *
 * 			 This is a C++ implmentation of https://github.com/astropy/astropy
 *
 *
 * @tparam:	vector<T> 	times:							times corresponding to data in values
 * @tparam:	vector<T1>	values:							measurement values
 * @param:	vector<double>	false_alarm_probability:	The false alarm probability (0 < fap < 1).
 * @param: optional<string>	method:						default='baluev' The approximation method to use;  options are 'baluev', 'davies', 'naive', 'bootstrap'
 * @param optional<double>	samples_per_peak:			defualt = 5: The approximate number of desired samples across the typical peak
 * @param optional<double>	nyquist_factor:				defualt = 5: The multiple of the average nyquist frequency used to choose the maximum frequency if maximum_frequency is not provided.
 * @param optional<double>	maximum_frequency:			If specified, then use this maximum frequency rather than one chosen based on the average nyquist frequency.
 *
 * @return map<double, double>		return:	Key = false alarm probability, value= false alarm level for respectve probability  The periodogram peak height corresponding to the specified false alarm probability.
 *
 *  Notes
        -----
        The true probability distribution for the largest peak cannot be
        determined analytically, so each method here provides an approximation
        to the value. The available methods are:

        - "baluev" (default): the upper-limit to the alias-free probability,
          using the approach of Baluev (2008) [1]_.
        - "davies" : the Davies upper bound from Baluev (2008) [1]_.
        - "naive" : the approximate probability based on an estimated
          effective number of independent frequencies.
        - "bootstrap" : the approximate probability based on bootstrap
          resamplings of the input data.

        Note also that for normalization='psd', the distribution can only be
        computed for periodograms constructed with errors specified.

        References
        ----------
        .. [1] Baluev, R.V. MNRAS 385, 1279 (2008)
 *
 *
 */

template<typename T, typename T1>
std::map<double, double> false_alarm_level(std::vector<T> &times, std::vector<T1> &values, std::vector<double> &false_alarm_probability, std::experimental::optional<std::string> method,std::experimental::optional<double> samples_per_peak,
		std::experimental::optional<double> nyquist_factor, std::experimental::optional<double> minimum_frequency,std::experimental::optional<double> maximum_frequency){


	method = method.value_or("baluev");
	samples_per_peak = samples_per_peak.value_or(5.0);

	nyquist_factor= nyquist_factor.value_or(5.0);
	minimum_frequency=minimum_frequency.value_or(0.0);
	maximum_frequency=maximum_frequency.value_or(0.00);
	std::string normalization = "standard";

	std::vector<double> my_min_max({0.0000,0.00000});
	double fmax=0.00;
	double val_returned = 0.000;
	std::map<double, double> false_alarm_levels_val;


	if(*maximum_frequency ==0){
		my_min_max = Stats::AutoFrequency<double,T>(times,samples_per_peak ,nyquist_factor,minimum_frequency,maximum_frequency, true);
		fmax = my_min_max[1];
	}
	else{
		fmax = *maximum_frequency;

	}


	for(auto i:false_alarm_probability){

		if(method.value().compare("baluev") == 0)
			val_returned = Stats::inv_fap_baluev<double,double>(i,fmax, times,values,normalization);

		false_alarm_levels_val[i] = val_returned;

	}

	return false_alarm_levels_val;


}
template std::map<double, double> false_alarm_level(std::vector<double> &, std::vector<double> &, std::vector<double> &, std::experimental::optional<std::string> ,std::experimental::optional<double> ,
		std::experimental::optional<double> , std::experimental::optional<double> ,std::experimental::optional<double> );




/*
 * @brief: Struct used to hold data for passing to functions when using cminpack
 *
 */
template<typename T, typename T1>
struct LombScargleFAP{
	std::vector<T> *t;
	std::vector<T1> *y;
	double fmax;
	double false_alarm_probability;

};

int fcn( void *p , int M, int N,const double *X, double *FVEC, int IFLAG){


	struct Stats::LombScargleFAP<double, double> *ls = (struct Stats::LombScargleFAP<double,double> *)p;

	double x_ =0.0;
	double fmax_ = ls->fmax;
	std::vector<double> t;
	std::vector<double> y;

	FVEC[0] = Stats::fap_baluev<double,double>(X[0],fmax_,*ls->t, *ls->y, 1, "standard")-ls->false_alarm_probability;

return 0;

};

/*
 * @brief:	Inverse of the Baluev alias-free approximation
 * 			This is a C++ implmentation of https://github.com/astropy/astropy
 * 
 * @param 	double		 fap:				The false alarm probability (0 < fap < 1).
 * @param 	double 		maximum_frequency:	If specified, then use this maximum frequency rather than one chosen based on the average nyquist frequency.
 * @tparam	vector<T> 	times				times corresponding to data in values
 * @tparam	vector<T1>	values:				measurement values
 * @param	string		normalization:		default = standard: standardize the periodogram
 * 
 */

template<typename T, typename T1>
double inv_fap_baluev(double fap, double maximum_frequency, std::vector<T> &times, std::vector<T1> &values, std::string normalization){
    //"""Inverse of the Baluev alias-free approximation"""
  
   double z0 = 0.0000000;
   z0 =Stats::inv_fap_naive<double,double>(fap,maximum_frequency,times,normalization);

   LombScargleFAP<double, double> what;

   what.t = &times;
   what.y = &values;
   what.fmax = maximum_frequency;
   what.false_alarm_probability = fap;

   int m=1;//times.size();
   int n= 1;
   double x[n];
   double fvec[m];
   double ftol = 1.49012e-8;
   double xtol = 1.49012e-8;
   double gtol = 0.0;
   int maxfev = -10;
   double epsfcn = 0.0;
   double diag[n];
   int mode = 2;
   double factor = 1.0e2;
   int nprint = 0;
   int info = 0;
   int nfev;
   double fjac[m*n];
   int ldfjac;
   int ipvt[n];
   double qtf[n];
   double wa1[n];
   double wa2[n];
   double wa3[n];
   double wa4[m];

   if (maxfev < 0)
	   maxfev = 200*(n+1);

   x[0]= z0;
   fvec[0]= z0;

   info = __cminpack_func__(lmdif)(fcn, &what, m, n, x, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor, nprint, &nfev, fjac,
		   ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4);


   return x[0];
}
template double inv_fap_baluev<double,double>(double , double , std::vector<double> &t, std::vector<double> &y, std::string normalization);

/*
 * @brief:	Inverse FAP based on estimated number of indep frequencies
 * 			This is a C++ implmentation of https://github.com/astropy/astropy
 *
 * @param 	double		 fap:				The false alarm probability (0 < fap < 1).
 * @tparam	vector<T> 	times				times corresponding to data in values
 * @param	string		normalization:		default = standard: standardize the periodogram
 *
 */
template<typename T, typename T1>
double inv_fap_naive(double fap, double fmax, std::vector<T> &t, std::string normalization){
    //Inverse FAP based on estimated number of indep frequencies
    //fap = np.asarray(fap)
    int N = t.size();

    auto [my_min, my_max] = std::minmax_element(std::begin(t), std::end(t));
    T my_T = my_max - my_min;
    double N_eff = fmax * my_T;

    long double fap_s = -std::expm1(std::log(1 - fap) / N_eff);
    return Stats::inv_fap_single(fap_s, N, normalization);

}
template double inv_fap_naive<double,double>(double, double, std::vector<double> &, std::string);

double inv_fap_single(double fap, int N, std::string normalization, int dH, int dK){
    /*"""Single-frequency inverse false alarm probability

    This function computes the periodogram value associated with the specified
    single-frequency false alarm probability. This should not be confused with
    the false alarm level of the largest peak.

    Parameters
    ----------
    fap : array_like
        The false alarm probability.
    N : int
        The number of data points from which the periodogram was computed.
    normalization : {'standard', 'model', 'log', 'psd'}
        The periodogram normalization.
    dH, dK : integers, optional
        The number of parameters in the null hypothesis and the model.

    Returns
    -------
    z : np.ndarray
        The periodogram power corresponding to the single-peak false alarm
        probability.

    Notes
    -----
    For normalization='psd', the distribution can only be computed for
    periodograms constructed with errors specified.
    All expressions used here are adapted from Table 1 of Baluev 2008 [1]_.

    References
    ----------
    .. [1] Baluev, R.V. MNRAS 385, 1279 (2008)
    """*/

	//fap = np.asarray(fap)

//	make this a map or just return a vector in the order they were put in??

	if(dK - dH != 2)
		std::cout<<"Degrees of freedom != 2"<<std::endl;
	int Nk = N - dK;

	// # No warnings for fap = 0; rather, just let it give the right infinity.
	// with np.errstate(divide='ignore'):
	if(normalization.compare("psd")==0)
		return -std::log(fap);
	else if(normalization.compare("standard")== 0)
		return 1 - std::pow(fap ,(2.00 / Nk));
	else if(normalization.compare("model") == 0)
		return -1 + std::pow(fap ,(-2.00 / Nk));
	else if(normalization.compare("log") == 0)
		return -2 / Nk * std::log(fap);
	else{
		std::cout<<"normalization='{}' is not recognized "<< normalization<<std::endl;
		return -1.0000;
	}


}

template<typename T, typename T1>
double fap_baluev(double Z, double fmax, std::vector<T> &t, std::vector<T1> &y, double dy , std::string normalization){
    //"""Alias-free approximation to false alarm probability

    //(Eqn 6 of Baluev 2008)
    //"""

    double fap_s = Stats::fap_single(Z, t.size(), normalization);
    double tau = Stats::tau_davies(Z, fmax, t, y, dy, normalization);


//    # result is 1 - (1 - fap_s) * np.exp(-tau)
//    # this is much more precise for small numbers
    return ((-std::expm1(-tau) + fap_s * std::exp(-tau)));

}
template double fap_baluev<double, double>(double, double, std::vector<double> &, std::vector<double> &, double, std::string);



double fap_single(double z, int N, std::string normalization, int dH, int dK){
	/*"""Single-frequency false alarm probability for the Lomb-Scargle periodogram

	    This is equal to 1 - cdf, where cdf is the cumulative distribution.
	    The single-frequency false alarm probability should not be confused with
	    the false alarm probability for the largest peak.

	    Parameters
	    ----------
	    z : array_like
	        The periodogram value.
	    N : int
	        The number of data points from which the periodogram was computed.
	    normalization : {'standard', 'model', 'log', 'psd'}
	        The periodogram normalization.
	    dH, dK : integers, optional
	        The number of parameters in the null hypothesis and the model.

	    Returns
	    -------
	    false_alarm_probability : np.ndarray
	        The single-frequency false alarm probability.

	    Notes
	    -----
	    For normalization='psd', the distribution can only be computed for
	    periodograms constructed with errors specified.
	    All expressions used here are adapted from Table 1 of Baluev 2008 [1]_.

	    References
	    ----------
	    .. [1] Baluev, R.V. MNRAS 385, 1279 (2008)
	    """*/
	//z = np.asarray(z)
	if(dK - dH != 2)
		std::cout<<"Error Degrees of freedom != 2"<<std::endl;
	double Nk = (double)N - (double)dK;

	if(normalization.compare("psd") == 0)
		return std::exp(-z);
	else if(normalization.compare("standard") == 0)
		return std::pow((1 - z),(0.5 * Nk));
	else if(normalization.compare("model") == 0)
		return std::pow((1 + z),(-0.5 * Nk));
	else if(normalization.compare("log") == 0)
		return std::exp(-0.5 * Nk * z);
	else{
		std::cout<<"normalization="<<normalization<<" is not recognized"<<std::endl;
		return -1.0000;
	}


}


double _gamma(double N){

	return std::sqrt(2 / N) * std::exp(::lgamma(N / 2.0) - ::lgamma((N - 1.0) / 2.0));
}


template<typename T, typename T1>
double tau_davies(double Z, double fmax, std::vector<T> &t, std::vector<T1> &y, double dy, std::string normalization, double dH, double dK){
    //"""tau factor for estimating Davies bound (Baluev 2008, Table 1)"""
    int N = t.size();
    double NH = N - dH;//# DOF for null hypothesis
    double NK = N - dK;//# DOF for periodic hypothesis

    bacc::accumulator_set<T, bacc::stats< bacc::tag::weighted_mean, bacc::tag::mean>, T> acc;
    bacc::accumulator_set<T, bacc::stats< bacc::tag::weighted_mean, bacc::tag::mean>, T> acc_squared;

    for(auto i:t){
    	acc(i, boost::accumulators::weight = dy);
    	acc_squared(std::pow(i,2),boost::accumulators::weight = dy);
    }

    double Dt = bacc::weighted_mean(acc_squared)-std::pow(bacc::weighted_mean(acc),2);//_weighted_var(t, dy);

    double my_pi=3.1415926535897932384626433832795029l;

    double Teff = std::sqrt(4 * my_pi * Dt);// # Effective baseline
    double W = fmax * Teff;
    //Z = np.asarray(Z)
    if(normalization.compare("psd")==0)
    	//# 'psd' normalization is same as Baluev's z
    	return W * std::exp(-Z) * std::sqrt(Z);
    else if(normalization.compare("standard")==0)
    	//# 'standard' normalization is Z = 2/NH * z_1
    	return (Stats::_gamma(NH) * W * std::pow((1 - Z),(0.5 * (NK - 1)) )
    			* std::sqrt(0.5 * NH * Z));
    else if(normalization.compare("model")==0)
    	//# 'model' normalization is Z = 2/NK * z_2
    	return (Stats::_gamma(NK) * W * std::pow((1 + Z), (-0.5 * NK))
    			* std::sqrt(0.5 * NK * Z));
    else if(normalization.compare("log")==0)
    	//# 'log' normalization is Z = 2/NK * z_3
    	return (Stats::_gamma(NK) * W * std::exp(-0.5 * Z * (NK - 0.5))
    			* std::sqrt(NK * std::sinh(0.5 * Z)));
    else
    	return -1.0000;


}
template double tau_davies<double, double>(double, double, std::vector<double> &, std::vector<double> &, double, std::string,double,double);







/*
 * @briref:	wrapper for gretl shapiro_wilk test for normality; iterates over a map<string, vector<double> >
 * 			set transform to true if you want do log(1+x) were log=ln on the data
 *
 * @prarm: std::map<std::string, std::vector<double> > data: data to test for normality
 * @param: bool transform: perform ln(1+x) tranform on the data
 *
 */
#if GRETL == 1

void SetGretlPaths(std::string path){

	int err;
	ConfigPaths config_path;
	PRN *prn;

	char gnu_dir[]="/usr/local/bin/gnuplot";
	strcpy(config_path.gnuplot ,gnu_dir);

	std::string gertl_tmp = path+"GretlTmp";
	strcpy(config_path.workdir ,const_cast<char*>(gertl_tmp.c_str()));

	err = gretl_set_paths (&config_path);

	if (err) {
		errmsg(err, prn);
		exit(EXIT_FAILURE);
	}


}

void ShapiroWilksGretl(std::map<std::string, std::vector<double> > &data, bool transform){
	//Test for normality
	std::map<std::string, std::vector<double> >::iterator data_iter = data.begin();
	double w= 0;
	double p=0;
	std::vector<double> transformed;

	if(!transform){
		for(; data_iter != data.end(); ++data_iter){

			if(::shapiro_wilk(data_iter->second.data(), 0, data_iter->second.size()-1, &w,&p ) == 0 ){

				if(p>.05)
					std::cout<<"Normal Distributions "<<data_iter->first<<" W "<<w<<"P value: "<<p<<std::endl;
			}
			else{
				std::cout<<"Shapiro Wilkes Test Failed "<<data_iter->first<<" Size of data set "<<data_iter->second.size()-1<<std::endl;
				for(uint i =0; i<data_iter->second.size()-1; ++i){
					if(data_iter->second[i]>0)
						std::cout<<data_iter->second[i]<<std::endl;
				}
			}
		}
	}
	else{
		for(; data_iter != data.end(); ++data_iter){
			transformed.resize(data_iter->second.size());
			for(uint i=0; i<transformed.size(); ++i)
				transformed[i] = std::log1p(data_iter->second[i]);

			if(::shapiro_wilk(transformed.data(), 0, transformed.size()-1, &w,&p ) == 0 ){

				if(p>.05)
					std::cout<<"Normal Distributions "<<data_iter->first<<" W "<<w<<"P value: "<<p<<std::endl;
			}
			else{
				std::cout<<"Shapiro Wilkes Test Failed "<<data_iter->first<<" Size of data set "<<data_iter->second.size()-1<<std::endl;
				for(uint i =0; i<data_iter->second.size()-1; ++i){
					if(data_iter->second[i]>0)
						std::cout<<data_iter->second[i]<<std::endl;
				}
			}
		}//end Test for normality
	}
}

void ShapiroWilksGretl(DATASET *dset, std::vector<double> variables, bool transform){
	//Test for normality
	double w= 0;
	double p=0;
	std::vector<double> transformed;

//	if(!transform){
//		for(auto variable_iter : variables){
//
//			if(::shapiro_wilk(dset->Z[variable_iter], 0, dset->n, &w,&p ) == 0 ){
//
//				if(p>=.05)
//					std::cout<<"Normal Distributions "<<variable_iter<<" W "<<w<<"P value: "<<p<<std::endl;
//				else
//					std::cout<<"Not Normal Distributions "<<variable_iter<<" W "<<w<<"P value: "<<p<<std::endl;
//			}
//			else{
//				std::cout<<"Shapiro Wilkes Test Failed "<variable_iter<<" Size of data set "<<dset->n<<std::endl;
//				/*for(uint i =0; i<data_iter->second.size()-1; ++i){
//					if(data_iter->second[i]>0)
//						std::cout<<data_iter->second[i]<<std::endl;
//				}*/
//			}
//		}
//	}
//	else{
//		for(auto variable_iter : variables){
//			transformed.resize(dset->n);
//			for(uint i=0; i<transformed.size(); ++i)
//				transformed[i] = std::log1p(dset->Z[*variable_iter][i]);
//
//			if(::shapiro_wilk(transformed.data(), 0, transformed.size()-1, &w,&p ) == 0 ){
//
//				if(p>=.05)
//					std::cout<<"Normal Distributions "<<variable_iter<<" W "<<w<<"P value: "<<p<<std::endl;
//				else
//					std::cout<<"Not Normal Distributions "<<variable_iter<<" W "<<w<<"P value: "<<p<<std::endl;
//			}
//			else{
//				std::cout<<"Shapiro Wilkes Test Failed "<variable_iter<<" Size of data set "<<dset->n<<std::endl;
//				/*for(uint i =0; i<data_iter->second.size()-1; ++i){
//					if(data_iter->second[i]>0)
//						std::cout<<data_iter->second[i]<<std::endl;
//				}*/
//			}
//		}//end Test for normality
//	}
}


/*
 * @brief: perform an arima
 *
 * @param DATASET	*dset:	GRETL data set
 * @param PRN		*prn: 	GRETL print data set
 * @param MODEL		*model:	GRETL model data set
 * @param int		p:		AR order
 * @param int		d:		order of integreation
 * @param int		q:		MA order
 * @param int		position_of_depent_variable: the position in dset->Z of variable
 * @param int		use_residuals = 0:	if = 1 then will put the residual in the the DATASET
 *
 */
int ArmaEstimateGretl (DATASET *dset, PRN *prn, MODEL *model, int p, int d, int q, int position_of_dependent_variable, int *use_residuals)
{
	;
	int *list;
	int err;

	model = gretl_model_new();
	list = gretl_list_new(5);

	list[1] = p;        /* AR order */
	list[2] = d;        /* order of integration */
	list[3] = q;        /* MA order */
	list[4] = LISTSEP;  /* separator */
	list[5] = position_of_dependent_variable;        /* position of dependent variable in dataset */

	*model = arma(list, NULL, dset, OPT_NONE, prn);
	err = model->errcode;

	if (err) {
		errmsg(err, prn);
	} else {
		printmodel(model, dset, OPT_NONE, prn);
	}

	if (!err && *use_residuals) {
		/* save arma residual series? */
		int v;

		dataset_add_allocated_series(dset, model->uhat);
		v = dset->v - 1 ; /* ID number of last series */
		strcpy(dset->varname[v], "residual");
		*use_residuals = v;
		model->uhat = NULL; /* don't double free! */
	}

	gretl_model_free(model);
	free(list);

	return err;
}

/*
 * @brief: call gretl to calculate and plot hurst exponent
 *
 * @param const int		*list:		list of variable index
 * @param DATASET		*dset:		Gretl DATASET containg the data
 * @param gretlopt		*opt:		Gretl option
 * @param PRN			*prn:		Gretl print pointer
 * @param bool			no_plot:	true == display plot of hurst exponents
 *
 */
double HurstExponentGretl(const int *list, DATASET *dset, gretlopt opt,PRN *prn, bool no_plot, std::string path_name){

	int err = 0;
	double my_hurst_exponent = -1.00000;
	gretl_matrix *hurst_mat;// = NULL;
	GretlType *hurst_ = new GretlType;

	if(!no_plot){
		if(path_name.compare("-1") != 0){
			path_name.append("_hurst.png");
			set_optval_string(::HURST, OPT_U, path_name.c_str());
		}
		err = hurstplot(list, dset, opt,prn);
	}

	hurst_mat = reinterpret_cast<gretl_matrix *>(get_last_result_data (hurst_, &err));
	my_hurst_exponent = hurst_mat->val[0];

	gretl_matrix_free(hurst_mat);

	delete hurst_;

	return my_hurst_exponent;

}

/*
 * @breif: computes, plots, and returns the values of ACF and PACF
 *
 * @param int			varno: 		index location of variable in DATASET that is being tested
 * @param int			max_lag:	max number of lags
 * @param DATASET		*dset:		Gretl DATASET containg the data
 * @param gretlopt		*opt:		Gretl option
 * @param PRN			*prn:		Gretl print pointer
 *
 * @return std::pair<std::map<double, int>,std::map<double, int> > acf_pacf_values: acf_pacf_values.first = ACF and acf_pacf_values.second = PACF,
 * 																					were map.key = lag, and map.value = acf/pacf value
 *
 */
std::pair<std::map<double, int>,std::map<double, int> > CorrgramGretl(int varno, int max_lag, DATASET *dset, gretlopt opt, PRN *prn, std::string path_name){

	int err = 0;
	gretl_matrix *cmat;

	/*lag*/ /*value*/
	std::map<double, int> acf_values;
	std::map<double, int> pacf_values;

	std::pair<std::map<double, int>,std::map<double, int> > acf_pacf_values;

	/* call correlogram_plot() directly */
	cmat = acf_matrix(dset->Z[varno], max_lag, dset, dset->n, &err);

	if (!err) {
		/* take pointers into the two @cmat columns */
		const double *acf = cmat->val;
		const double *pacf = acf + cmat->rows;
		/* set the plus/minus range */
		double pm = 1.96 / sqrt(dset->n);

		for(int i = 0; i<2*cmat->rows; ++i){
			if(i<cmat->rows)
				acf_values[cmat->val[i]] = i;
			if(i>=cmat->rows){
				pacf_values[cmat->val[i]] = i-cmat->rows;
			}
		}
		if(path_name.compare("-1") != 0){
			path_name.append("_correlogram.png");
			set_optval_string(::CORRGM, OPT_U, path_name.c_str());
		}
		correlogram_plot(dset->varname[varno], acf, pacf, NULL,
				max_lag, pm, opt);
	}
	gretl_matrix_free(cmat);


	if (err) {
		errmsg(err, prn);
	}

	acf_pacf_values = std::make_pair(acf_values,pacf_values);

	return acf_pacf_values;
}


/*
 * @brief: computes and plots periodogram
 *
 * @param int			varno: 		index location of variable in DATASET that is being tested
 * @param int			width:		width size when using bartlet window
 * @param DATASET		*dset:		Gretl DATASET containg the data
 * @param gretlopt		*opt:		Gretl option
 * 									OPT_O == bartlet
 * 									OPT_L == ln
 * @param PRN			*prn:		Gretl print pointer
 * @param bool			no_plot:	true == display plot and out put spectral_density_period values
 * @param string		path_name:	default = -1, saves plot if to value of path_name if provided
 *
 * @return std::map<double, double>		spectral_density_period .first == sprectral density, .second = period
 *
 */
std::map<double, double> PeriodogramGretl(int varno, int width, DATASET *dset, gretlopt opt, PRN *prn, bool no_plot, std::string path_name){


	int err;
	gretl_matrix *pmat;
	int T = dset->t2 - dset->t1 + 1;


	/*spectral_density period*/
	std::map<double,double> spectral_density_period;


	if(!no_plot){
		if(path_name.compare("-1") != 0){
			path_name.append("_periodogram.png");
			set_optval_string(::PERGM, OPT_U, path_name.c_str());
		}
		periodogram(varno, width, dset, opt, prn);
	}

	if(opt == OPT_O){
		if(width < 0)//see https://gretlml.univpm.it/hyperkitty/list/gretl-users@gretlml.univpm.it/thread/ADMUPRNQYE4N6AHQBK3QCSO57MBX7464/
			width = 2*sqrt(dset->t2);
	}
	if(opt != OPT_O)
		width = -1;

	//this only does OPT_NONE or OPT_0
	pmat = periodogram_matrix (*dset->Z,dset->t1,dset->t2, width , &err);

	if(opt == OPT_L){
		for(int i = 0; i<pmat->rows; ++i)
			spectral_density_period[std::log(pmat->val[i+pmat->rows])] = T/((double)i+1);

	}
	else{
		for(int i = 0; i<pmat->rows; ++i)
			spectral_density_period[pmat->val[i+pmat->rows]] = T/((double)i+1);

	}

	if(no_plot){
		for(auto i : spectral_density_period)
			std::cout<<i.first<<" "<<i.second<<std::endl;
	}

	gretl_matrix_free(pmat);


	if (err) {
		errmsg(err, prn);
	}


	return spectral_density_period;

}


Summary *SummaryGretl(const DATASET *dset, PRN *prn, int *err){

	Summary *summary;
	gretlopt opt = OPT_NONE;//OPT_U;//OPT_B;
	//int list1[2] = {1,0};

	int list1[dset->v+1];

	list1[0] = dset->v;
	for(int i=1; i<=dset->v; ++i)
		list1[i] = i-1;

	summary = get_summary(list1,dset, opt,prn,err);

	print_summary (summary, dset,prn);

	return summary;


}

/*
 * @brief Creates a Gretl DATASET, Performs Exploratory data Analysis and returns a gretl Summary *
 *
 * @tparam &data:					the container that holds the data for EDA
 * @param std::vector<std::string> &gretl_variable_name: vector of variables names
 * @param bool is_diffed: True is the data is already differed, false otherwise default = false
 *
 */
template<typename T>
Summary *EDA(std::string path,  T &data, std::vector<std::string> &gretl_variable_name, std::string additional_name, bool is_diffed){

	std::cout<<"Stats::EDA"<<std::endl;

	//for(auto it:data){



		// int use_residual = 0; /* or 0 */
//		int vnum=0;             /* series ID for correlogram */
//		gretlopt opt = OPT_NONE;
		DATASET *dset;
//		DATASET *diff_dset;
		PRN *prn;
//		MODEL *model;
		Summary *summary;
		gretlopt opt = OPT_NONE;//OPT_U;//OPT_B;
		//int list[12] = {11,1,2,3,4,5,6,7,8,9,10,11};
		//int list[2] = {1,0};
		int list[gretl_variable_name.size()+1];//
		list[0] = gretl_variable_name.size();
		for(int i = 1; i<=list[0]; ++i)
			list[i] = i-1;

//		int list1[2] = {1,0};
		int err=0;





//		ConfigPaths config_path;

//		gretl_matrix *cmat;
//		gretl_matrix *pmat;

		std::string box_plot_path = path+"GretlTmp/gpttmp.plt";
		const char *literal = box_plot_path.c_str();//"test";
//		int use_residuals =0;
//		int max_lag = 365;//what about seasonal products thus mostly zero's


		dset = datainfo_new();

		prn = gretl_print_new(GRETL_PRINT_STDOUT, NULL);

		std::string name_path = path;





		//std::vector<std::string> var_names=this->gretl_variable_name;//({"constant","2011","2012","2013","2014", "2015","2016","2017", "2018","2019","2020", "2021"});


		MyData::ToGretl<double>(data,dset,7,1,gretl_variable_name, false);//MyData::ToGretl(data,dset,7,1, "test");

//		double p_value = 0.000;


/*			//start: adf kpss tests
			{
				//DATASET *dset1;
				//int list2[3] = {2,0,1};
				//dset1 = datainfo_new();
				//MyData::ToGretl(data,dset,7,1, var_name, true);


				//int list3[3] = {2,1,1};
				int adf_error = ::adf_test(1,list,dset,OPT_T,prn);

				if (adf_error!=0) {
					errmsg(adf_error, prn);
				}

				int kpss_error = ::kpss_test(1,list,dset,OPT_T,prn);

				if (kpss_error!=0) {
					errmsg(kpss_error, prn);
				}

				int plist[2] = {1,3};
				int levin_error = ::levin_lin_test(1, plist, dset,OPT_NONE,prn);

				if (levin_error!=0) {
					errmsg(levin_error, prn);
				}

				int fractint_error = ::fractint(1,25,dset,OPT_A,prn);

				if (fractint_error!=0) {
					errmsg(fractint_error, prn);
				}

				//p_value = get_last_pvalue();//::get_urc_pvalue ();

			}
			//end: adf, kpss tests
			*/

		//H0: the sequence was produced in a random manner
		//for(uint i = 0; i<var_names.size(); ++i)
//		for(int i = list[1]; i<=list[0]; ++i){
//			std::cout<<"Variable "<<this->variable_idx_to_name[i]<<std::endl;
//			err = runs_test (i, dset,OPT_D, prn);
//		}
//
//		std::vector<double> gretl_noramlity_test_p_values(var_names.size());
//
//			for(int i = list[1]; i<=list[0]; ++i){
//				std::cout<<"Variable "<<this->variable_idx_to_name[i]<<std::endl;
//				::gretl_normality_test(list[i],dset, OPT_W,prn);
//				gretl_noramlity_test_p_values[i]=::get_last_pvalue();
//			}
//
//			if(is_diffed){
//				std::vector<std::vector<double> > tmp(1,gretl_noramlity_test_p_values);
//				ReadWrite::SaveDataVector<double, const std::vector<std::vector<double> > >(path,"ShapiroWilks_test_Year_diffed",tmp);
//			}
//			else{
//				std::vector<std::vector<double> > tmp(1,gretl_noramlity_test_p_values);
//				ReadWrite::SaveDataVector<double, const std::vector<std::vector<double> > >(path,"ShapiroWilks_test_Year",tmp);
//			}
//
//
		for(int i = list[1]; i<list[0]; ++i){

			name_path = path;
			if(additional_name.compare("none") == 0){
				if(is_diffed)
					name_path += gretl_variable_name[i]+"_hist_plot_diffed.png";
				else
					name_path += gretl_variable_name[i]+"_hist_plot.png";
			}
			else{
				if(is_diffed)
					name_path += gretl_variable_name[i]+additional_name+"_hist_plot_diffed.png";
				else
					name_path += gretl_variable_name[i]+additional_name+"_hist_plot.png";

			}

			set_optval_string(::FREQ, OPT_U, name_path.c_str());

			std::cout<<"Variable "<<gretl_variable_name[i]<<std::endl;
			//int have_freq = ::freqdist (i, dset,OPT_NONE,prn);//::freqdist (i, dset,OPT_Z,prn);OPT_Z plots normal curve
			//std::cout<<"do we have a frequencies "<<have_freq<<std::endl;
		}

		name_path = path;

		if(is_diffed)
			name_path += gretl_variable_name[0] + "box_plot_diff.png";
		else
			name_path += gretl_variable_name[0] + "box_plot.png";

		set_optval_string(::BXPLOT, OPT_U, name_path.c_str());

		std::map<boost::posix_time::ptime, ::Summary *> gretl_summary_vector;

		summary = Stats::SummaryGretl(dset, prn, &err);//get_summary (list1,dset, opt,prn,&err);//get_summary (list1,dset, opt,prn,&err);

		int *erf = &err;

		err = boxplots(list, literal,dset, OPT_U);

		boxplot_numerical_summary (literal, prn);
//
//		//  plots acf and pacf of the variable
//		//		err = Stats::ArmaEstimateGretl(dset,prn,model,14,0,1,0,&use_residuals);
//
//
//		std::pair<std::map<double, int>,std::map<double, int> > acf_pacf_values;
//		//for(uint i = 0; i<var_names.size(); ++i)
//		for(int i = list[1]; i<=list[0]; ++i){
//			name_path = path+var_names[i];
//			acf_pacf_values = Stats::CorrgramGretl(i, max_lag, dset, opt, prn, name_path);
//		}
//
//		Stats::AnalyzeACFPACF(acf_pacf_values, dset->n);
//
//		std::map<double,double> spectral_density_period;
//
//		//I want to use width = -1 and OPT_NONE
//		//		spectral_density_period = Stats::PeriodogramGretl(0,-1,dset,OPT_NONE, prn, true);
//
//
//		//	double my_hurst_exponent = (double)hurstplot(list, dset, opt,prn);
//
//		double _my_hurst_exponent = 0.00000;
//		for(int i = list[1]; i<=list[0]; ++i){
//			//std::cout<<std::endl;
//			name_path = path+var_names[i];
//			list1[1] = i;
//			std::cout<<"Variable "<<this->variable_idx_to_name[i]<<std::endl;
//			_my_hurst_exponent = Stats::HurstExponentGretl(list1, dset, opt,prn, false, name_path);
//		}



		//for(int i = 0; i< var_names.size(); ++i){
//		for(int i = list[1]; i<=list[0]; ++i){
//
//			//double gg_ray[dset->t2];
//			std::vector<double> gg_ray;//(data[i].size());
//			if(is_diffed){
//				//gg_ray = data[i-1];
//				std::cout<<"need to fix above"<<std::endl;
//			}
//			else{
//				//gg_ray.resize(data[i-1].size());
//				err = diff_series(dset->Z[i],gg_ray.data(),295,dset);
//			}
//
//			opt = OPT_F;
//			//diff_series(*dset->Z,g_ray.data(),opt,dset);
//			//std::cout<<::function_from_string("diff")<<std::endl;
//
//
//			std::cout<<var_names[i]<<std::endl;
//
//			//if (err) {
//			//		errmsg(err, prn);
//			//	}
//
//
//			//get these by month
//			int count = 0;
//			for(int i = 1; i<gg_ray.size(); ++i){
//				if(gg_ray[i] >0)
//					++count;
//
//			}
//			std::cout<<"%Up Days "<<(double)count/gg_ray.size()<<std::endl;
//
//			//std::vector<double> g_ray(gg_ray.begin(),gg_ray.end());
//			//		free_summary(summary);
//			//destroy_dataset(dset);
//
//			//DATASET *diff_dset;
//
//			//diff_dset = ::create_new_dataset(1,gg_ray.size(), 0);
//			//diff_dset = datainfo_new();
//			//dset = datainfo_new();
//			diff_dset = datainfo_new();
//			MyData::ToGretl(gg_ray,diff_dset,7,1, "diff_test");
//
//			//H0: the sequence was produced in a random manner
//			//data must be differed first
//			err = runs_test (0, diff_dset,OPT_E, prn);
//			err = runs_test (0, diff_dset,OPT_NONE, prn);
//
//
//			std::pair<std::map<double, int>,std::map<double, int> > diff_acf_pacf_values;
//			opt = OPT_NONE;
//			//diff_acf_pacf_values = Stats::CorrgramGretl(vnum, max_lag, diff_dset, opt, prn);
//			diff_acf_pacf_values = Stats::CorrgramGretl(0, max_lag, diff_dset, opt, prn);
//			::clear_datainfo(diff_dset,::CLEAR_FULL);
//
//		}
//
//
//		std::vector<std::vector<int> > combinations = Stats::Combinations({0,1,2,3,4,5,6,7,8,9,10}, 2);
//
//		std::vector<double> tmp_vect(11);
//		//for_each(tmp_vect.begin(),tmp_vect.end(), [](double i){*i = std::nan});
//		/*		std::fill(tmp_vect.begin(),tmp_vect.end(),std::nan(""));
//		std::vector<std::vector<double> > f_p_values_vect(var_names.size(), tmp_vect);
//		std::vector<std::vector<double> > t_p_values_vect(var_names.size(), tmp_vect);
//		std::vector<std::vector<double> > wilcoxon_p_values_vect(11, tmp_vect);*/
//
//		std::vector<std::vector<double> > f_p_values_vect(list[0], tmp_vect);
//		std::vector<std::vector<double> > t_p_values_vect(list[0], tmp_vect);
//		std::vector<std::vector<double> > wilcoxon_p_values_vect(list[0], tmp_vect);
//
//		for(auto i:combinations){
//			//::diff_test(i.data(), dset,)
//			std::cout<<"Means Test for "<<var_names[i[0]]<<" , "<<var_names[i[1]]<<std::endl;
//
//			int list2[3] = {2, i[0], i[1]};
//			::vars_test(list2,dset,prn);
//			double p_value = ::get_last_pvalue();
//
//			f_p_values_vect[i[0]][i[1]] = p_value;
//
//
//			if(p_value<=.10)
//				::means_test(list2,dset,::OPT_O,prn);
//			else
//				::means_test(list2,dset,::OPT_NONE,prn);
//			p_value = ::get_last_pvalue();
//			t_p_values_vect[i[0]][i[1]] = p_value;
//
//			if(p_value<=.10)
//				std::cout<<"t-test sig"<<std::endl;
//
//			::diff_test(list2,dset,::OPT_R, prn);
//
//			p_value = ::get_last_pvalue();
//			wilcoxon_p_values_vect[i[0]][i[1]] = p_value;
//			if(p_value<=.10)
//				std::cout<<"Wilcoxon sig"<<std::endl;
//
//
//		}
//
//		if(is_diffed){
//			ReadWrite::SaveDataVector<double, const std::vector<std::vector<double> > >(path,"F_test_Year_diffed",f_p_values_vect);
//			ReadWrite::SaveDataVector<double, const std::vector<std::vector<double> > >(path,"T_test_Year_diffed",t_p_values_vect);
//			ReadWrite::SaveDataVector<double, const std::vector<std::vector<double> > >(path,"Wilcoxon_test_Year_diffed",wilcoxon_p_values_vect);
//		}
//		else{
//			ReadWrite::SaveDataVector<double, const std::vector<std::vector<double> > >(path,"F_test_Year",f_p_values_vect);
//			ReadWrite::SaveDataVector<double, const std::vector<std::vector<double> > >(path,"T_test_Year",t_p_values_vect);
//			ReadWrite::SaveDataVector<double, const std::vector<std::vector<double> > >(path,"Wilcoxon_test_Year",wilcoxon_p_values_vect);
//		}

		/*
		 * and here also trying to figure out how to find peaks and valled and troughs in the acf and pacf functions
		 * I am trying to figure out lag without looking at the chart-> i.e have the program do it
		 * I got a start in CorrgramGretl know how or
		 *
		 */


		std::cout<<"gretl var names with dropped variables"<<std::endl;
		for(int i = 0; i< dset->v; ++i)
			::printf("%s\n",dset->varname[i]);


		//free_summary(summary);
		destroy_dataset(dset);
//		if(diff_dset != NULL)
//			destroy_dataset(diff_dset);
		gretl_print_destroy(prn);


		//}

		return summary;


}
template Summary *EDA<std::multimap<boost::posix_time::ptime/*time*/, std::vector<double/*level_details*/> > >(std::string, std::multimap<boost::posix_time::ptime/*time*/, std::vector<double/*level_details*/> >&, std::vector<std::string> &, std::string, bool );
template Summary *EDA<std::map<boost::posix_time::time_duration/*end time of time period*/, std::multimap<boost::posix_time::ptime/*time*/,std::vector<double/*order_details*/> > > >(std::string,
		std::map<boost::posix_time::time_duration/*end time of time period*/, std::multimap<boost::posix_time::ptime/*time*/,std::vector<double/*order_details*/> > >&, std::vector<std::string> &, std::string, bool );
template Summary *EDA<std::vector<std::vector<double> > >(std::string,std::vector<std::vector<double> > &, std::vector<std::string> &, std::string, bool );



/*
 * @brief puts gretl summary into a container
 *
 * @param std::vector<Summary*> summary_vect: 			vector containing the pointer to a gretl Summary
 * @param std::vector<std::string> &stats_to_get:		vector contianing the names of the simple stats to return
 * @param std::map<int, std::string> &variables_to_get:	key->variable index in gretl summary value->its corresponding gretl variable name
 *
 *
 */
std::map<std::string/*variable*/, std::map< std::string/*statistic*/, std::vector<double>/*results*/ > > PrepSummariesForSaving(std::vector<Summary*> summary_vect, std::vector<std::string> &stats_to_get, std::map<int, std::string> &variables_to_get){

	std::cout<<"PrepSummariesForSaving"<<std::endl;
	std::vector<double> tmp_vect(summary_vect.size());
	std::map< std::string, std::vector<double> > tmp_map;
	std::map<std::string, std::map< std::string, std::vector<double> > > variable_name_stat_values;
	std::map<int, std::map< std::string, std::vector<double> > > variable_name_stat_values_int;
	std::map<int, std::string>::iterator variables_to_get_iter;
	std::map<int, std::string> my_variables_to_get;
	set<int> keep, remove;


	std::map<int, int> var_to_get;

	//get summary with most missing variables
	int shortest_index = 0;
	int shortest_value = summary_vect[shortest_index]->list[0];
	for(uint j = 0; j < summary_vect.size(); ++j){
		for(int i = 1; i<=summary_vect[j]->list[0]; ++i){//iterates variables
			//std::cout<<"variable location in dset "<<summary_vect[j]->list[i]<<std::endl;
			if(shortest_value > summary_vect[j]->list[0]){
				shortest_index = j;
				shortest_value = summary_vect[shortest_index]->list[0];
			}
		}
	}

	for(auto iter: variables_to_get)
		var_to_get[iter.first] = 0;

	//count the number of summaries that contain the variable
	for(uint j = 1; j < summary_vect.size(); ++j){
		for(int i = 0; i<=summary_vect[j]->list[0]; ++i){
			var_to_get[summary_vect[j]->list[i]] +=1;
		}
	}

	int count= 0;
	int j_ = 0;
	for(int q = 1; q <= summary_vect[shortest_index]->list[0]; ++q/*,++count*/){
		auto i = var_to_get.find(summary_vect[shortest_index]->list[q]);
		int gretl_dset_variable_index = summary_vect[shortest_index]->list[q];

		variables_to_get_iter =variables_to_get.find(i->first);
		//if every summary contains the variable then procceed
		if(var_to_get[i->first] == summary_vect.size() -1){
			for(auto stat : stats_to_get){
				for(uint j = 0; j < summary_vect.size(); ++j){
					j_ = j;
					count = q;
					if( j != shortest_index ){
						count = 0;
						while(count < summary_vect[j]->list[0] && summary_vect[j]->list[count] != gretl_dset_variable_index)
							++count;
						if(j == 0 && count != q)
							std::cout<<"wha"<<std::endl;

					}
					if(summary_vect[j]->list[count] != gretl_dset_variable_index)
						continue;
					if(stat.compare("mean") == 0){
						tmp_vect[j] = summary_vect[j]->mean[count-1];
					}
					else if(stat.compare("median") == 0){
						tmp_vect[j] = summary_vect[j]->median[count-1];
					}
					else if(stat.compare("sd") == 0){
						tmp_vect[j] = summary_vect[j]->sd[count-1];
					}
					else if(stat.compare("low") == 0){
						tmp_vect[j] = summary_vect[j]->low[count-1];
					}
					else if(stat.compare("high") == 0){
						tmp_vect[j] = summary_vect[j]->high[count-1];
					}
					else if(stat.compare("iqr") == 0){
						tmp_vect[j] = summary_vect[j]->iqr[count-1];
					}
					else if(stat.compare("perc05") == 0){
						tmp_vect[j] = summary_vect[j]->perc05[count-1];
					}
					else if(stat.compare("perc95") == 0){
						tmp_vect[j] = summary_vect[j]->perc95[count-1];
					}
					else if(stat.compare("skew") == 0){
						tmp_vect[j] = summary_vect[j]->skew[count-1];
					}
					else if(stat.compare("xkurt") == 0){
						tmp_vect[j] = summary_vect[j]->xkurt[count-1];
					}
					else if(stat.compare("count") == 0){
						tmp_vect[j] = summary_vect[j]->n;
					}
					else if(stat.compare("missingcount") == 0){
						tmp_vect[j] = summary_vect[j]->misscount[count-1];
					}
				}
				variables_to_get_iter =variables_to_get.find(i->first);
				if(variables_to_get_iter != variables_to_get.end()){
					variable_name_stat_values[variables_to_get_iter->second][stat] = tmp_vect;
					tmp_vect[0]=-8675309;
					tmp_vect[1]=-8675309;
				}
			}
		}
	}

	return variable_name_stat_values;
}





#endif

/*
 *
 *
 */
void AnalyzeACFPACF(std::pair<std::map<double, int>,std::map<double, int> > acf_pacf_values, int size_data){

	double ci = 2/std::sqrt(size_data);//confidence interval

	int max_lag_above_ci_acf=-1;
	int max_lag_above_ci_pacf=-1;

	int min_lag_above_ci_acf=1;
	int min_lag_above_ci_pacf=1;

	std::map<double, int>::reverse_iterator acf_iter = acf_pacf_values.first.rbegin();
	std::map<double, int>::reverse_iterator pacf_iter= acf_pacf_values.second.rbegin();

	if(acf_pacf_values.first.size()>0 && acf_pacf_values.second.size()>0){
		double acf = acf_iter->first;
		double acf_lag = acf_iter->second;

		for(;acf_iter!=acf_pacf_values.first.rend();++acf_iter,++pacf_iter){
			if(acf_iter->first > ci){
				if(std::abs(acf_iter->second) > max_lag_above_ci_acf && acf_iter->second != 1){
					max_lag_above_ci_acf = acf_iter->second;
				}
				else if(std::abs(acf_iter->second) < min_lag_above_ci_acf){
					min_lag_above_ci_acf = acf_iter->second;
				}
			}

			if(pacf_iter->first > ci){
				if(std::abs(pacf_iter->second) > max_lag_above_ci_pacf && pacf_iter->second != 1){
					max_lag_above_ci_pacf = pacf_iter->second;
				}
				else if(std::abs(pacf_iter->second) < min_lag_above_ci_pacf){
					min_lag_above_ci_pacf = pacf_iter->second;
				}
			}
		}
	}
	else
		std::cout<<"Error AnalyzeACFPACF acf size "<<acf_pacf_values.first.size()<<" pacf size "<<acf_pacf_values.second.size()<<std::endl;



}


double PyTorchF1(torch::Tensor y_actual_tensor, torch::Tensor y_pred_tensor, bool is_training){

	int correct = 0;
	int zero_count = 0;
	int incorrect = 0;
	int tp = 0;
	int fn = 0;
	int fp = 0;
	int tn = 0;

	int p = 0;
	int n = 0;
	int pp =0;
	int pn = 0;

	double f1_score = 0.0000;
	torch::Tensor f1_score_tensor = torch::full({1},std::nan(""), torch::kDouble);

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> y_actual_unique = at::_unique2(y_actual_tensor,false,false, true);

	torch::Tensor y_pred = torch::full({y_pred_tensor.size(0)},std::nan(""), torch::kDouble);
//	std::cout<<y_pred<<std::endl;

	double positive_class = *std::get<0>(y_actual_unique)[0].data_ptr<double>();
	p = *std::get<2>(y_actual_unique)[0].data_ptr<long>();
	n = *std::get<2>(y_actual_unique)[1].data_ptr<long>();
	for(int i = 0; i< y_actual_tensor.sizes()[0]; ++i){
		y_pred[i] = *torch::argmax(y_pred_tensor[i]).data_ptr<long>();
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> y_pred_unique = at::_unique2(y_pred,false,false, true);


	double y = 0.000;
	double y_hat = 0.000;

	double precision = 0.000;
	double recall = 0.000;

	if(std::get<0>(y_actual_unique).size(0) == std::get<0>(y_pred_unique).size(0) ){
		if(std::get<0>(y_actual_unique).size(0) == 2){
			for(int i = 0; i< std::get<1>(y_actual_unique).size(0); ++i){
				y = *std::get<1>(y_actual_unique)[i].data_ptr<long>();
				y_hat = *std::get<1>(y_pred_unique)[i].data_ptr<long>();

				//std::cout<<"y "<<y<<" y_hat "<<y_hat<<std::endl;
				if(y == positive_class){
					if(y == y_hat)
						++tp;
					else
						++fn;
				}
				else{
					if(y == y_hat)
						++tn;
					else
						++fp;
				}
			}

		}
		else{
			//TODO: multiclass to be implemented
		}

		if(tp != 0){
			precision = (double)tp/((double)(tp+fp) );
			recall = (double)tp/((double)(tp+fn) );

			f1_score = (precision*recall/(precision+recall));
		}
		else
			f1_score = 0.0000;

		*f1_score_tensor[0].data_ptr<double>() = f1_score;



		//f1_score_tensor.requires_grad_(it_training);


	}
	else
		std::cout<<"PyTorchF1: std::get<0>(y_actual_unique).size(0) != std::get<0>(y_pred_unique).size(0)"<<std::endl;





return f1_score;

}








}//End: namespace Stats





























