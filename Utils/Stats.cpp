/*
 * Stats.cpp
 *
 *  Created on: Mar 18, 2019
 *      Author: ryan
 */

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

	std::cout.precision(5);
	for(auto key = stats.begin(); key != stats.end(); ++key){
		my_stats[key->first] = key->second();
		if(show_stats){
			cout<<key->first<<" "<<key->second()<<" ";
		}
	}
	if(show_stats)
		cout<<endl;


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


	//cv::waitKey(0);
}

template <typename T>
void PlotLine(std::vector<T> &data, string name){

	//std::vector<std::pair<float, float>> data;
	std::vector<float> values(data.begin(), data.end());

	//cvplot::Window::current("cvplot demo").offset({60, 100});

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




}





























