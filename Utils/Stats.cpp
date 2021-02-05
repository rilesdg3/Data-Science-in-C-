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

	int c = data.size();//data.size();//cache size for histogramm.

	std::map<std::string, double > my_stats;



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
		//why did I have this myAccumulator((double)data[j]);
		wh_acc((double)data[j]);
	}

	histogram_type hist = boost::accumulators::density(wh_acc);

	double total = 0.0;

	for( uint i = 0; i < hist.size(); i++ )
	{
		//std::cout << "Bin lower bound: " << hist[i].first << ", Value: " << hist[i].second << std::endl;
		hist_st.hist_vect_bin.push_back(hist[i].first);
		hist_st.hist_vect.push_back(hist[i].second);
		hist_st.marginals.push_back((float)hist[i].second);
		total += hist[i].second;
	}
	//std::cout << "Total: " << total << std::endl; //should be 1 (and it is)

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

	int c = data.size();

	hist_st.count = c;

	std::vector<int> sorted_index(data.size());//holds the index of data as if it were sorted, i.e. sorted_index[0] =12375 would be
	//either the min or max or whatever data[12375] sorted_index[1] = 896, would be the second min or max or whatever

	//std::vector<int> bin(data.size());
	hist_st.bin_index.resize(data.size());// = bin;
	std::iota(sorted_index.begin(), sorted_index.end(),0);
	std::sort(sorted_index.begin(), sorted_index.end(), [&data](T i1, T i2){return data[i1] < data[i2];} );
	stat_acc wh_acc(boost::accumulators::tag::density::num_bins = n_bins, boost::accumulators::tag::density::cache_size = c);

	//fill accumulator
	for (int j = 0; j < c; ++j)
	{
		//myAccumulator(data[j]);
		wh_acc(data[j]);
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
		for(; (it!=sorted_index.end() && data[*it] < hist[i].first); ++it){
			hist_st.bin_index[*it]=i;
			//cout<<"data[*it] < hist[i].first "<<(data[*it] < hist[i].first)<<endl;
			//cout<<"value "<<data[*it]<<" n_bin "<<hist_st.bin_index[*it]<<" *it "<<*it<< " Bin lower bound: " << hist[i].first<<endl;

		}

		total += hist[i].second;
	}

	//std::cout << "Total: " << total << std::endl; //should be 1 (and it is)

}
template void ComputeHistograms<double >(std::vector<double> & ,Histogram &, int);

template<typename T, typename T1>
void LaggedMI(std::vector<T> pred_vars, std::vector<T1> target, int n_bins, int min_lag, int max_lag, int lag_step){


	cout<<"LaggedMI start"<<endl;
	int first_min_index = 0;
	int min_index = 0;
	double first_min = std::numeric_limits<double>::max();
	double min = std::numeric_limits<double>::max();
	double mi = 0.0;
	bool found_first_min = false;
	int len = (max_lag-min_lag)/lag_step;
	std::vector<double > mi_values(len+1);
	int mi_values_it = 0;

	//std::vector<double > pred_vars_sub(pred_vars.begin(),pred_vars.end());
	//std::vector<double > target_sub(target.begin(),target.end());




	//cout<<"Entropy "<<Stats::Entropy(h1)<<endl;

	for(int i = min_lag; i<=max_lag; i=i+lag_step){
		std::vector<T > pred_vars_sub(pred_vars.begin(),pred_vars.end()-i);
		std::vector<T1> target_sub(target.begin()+i,target.end());

		Stats::Histogram h;
		Stats::Histogram h1;
		Stats::ComputeHistograms(target_sub, h1,n_bins);
		Stats::ComputeHistograms(pred_vars_sub, h,n_bins);
		mi=Stats::DiscreteMI(h,h1);
		cout<<"MI "<<mi<<endl;

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
			mi_values_it++;
		}


	}
	cout<<"LaggedMI end"<<endl;
	//PlotLine(mi_values, "LaggedMI");
	cout<<"First Min "<< first_min<<" Found at "<<first_min_index<<endl;
	cout<<"Min "<< min<<" Found at "<<min_index<<endl;

}
template void LaggedMI<float, double>(std::vector<float> pred_vars, std::vector<double> target, int n_bins, int min_lag, int max_lag, int lag_step);


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
		int n_bins_pred = pred_hist_st.marginals.size()-1;
		int n_bins_target = target_hist_st.marginals.size()-1;
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
 * @tparam:	vector<T> 	times:			times corresponding to data in values
 * @tparam:	vector<T1>	values:			measurement values
 * @tparam:	vector<T2>	frequencies:	frequencies for output of periodogram
 * @param:	bool 		center_data:	default = true: subtract the mean
 * @param:	string 		normalization:	default = standard: standardize the periodogram
 *
 *
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
std::vector<T> AutoFrequency(std::vector<T1> times, std::experimental::optional<double> samples_per_peak, std::experimental::optional<double> nyquist_factor, std::experimental::optional<double> minimum_frequency,
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
template std::vector<double> AutoFrequency<double, double>(std::vector<double> times, std::experimental::optional<double> samples_per_peak, std::experimental::optional<double> nyquist_factor, std::experimental::optional<double> minimum_frequency,
		std::experimental::optional<double> maximum_frequency,std::experimental::optional<bool> return_freq_limits);








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
double HurstExponentGretl(const int *list, DATASET *dset, gretlopt opt,PRN *prn, bool no_plot){

	int err = 0;
	double my_hurst_exponent = -1.00000;
	gretl_matrix *hurst_mat;// = NULL;
	GretlType *hurst_ = new GretlType;

	//if(!no_plot)
		err = hurstplot(list, dset, opt,prn);

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
std::pair<std::map<double, int>,std::map<double, int> > CorrgramGretl(int varno, int max_lag, DATASET *dset, gretlopt opt, PRN *prn){

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
 *
 * @return std::map<double, double>		spectral_density_period .first == sprectral density, .second = period
 *
 */
std::map<double, double> PeriodogramGretl(int varno, int width, DATASET *dset, gretlopt opt, PRN *prn, bool no_plot){


	int err;
	gretl_matrix *pmat;
	int T = dset->t2 - dset->t1 + 1;


	/*spectral_density period*/
	std::map<double,double> spectral_density_period;


	if(!no_plot)
		periodogram(varno, width, dset, opt, prn);

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

	if(!no_plot){
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
	int list1[2] = {1,0};

	summary = get_summary(list1,dset, opt,prn,err);

	print_summary (summary, dset,prn);

	return summary;


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










}





























