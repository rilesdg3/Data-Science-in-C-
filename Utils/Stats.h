/*
 * Stats.h
 *
 *  Created on: Mar 18, 2019
 *      Author: ryan
 */

#ifndef STATS_H_
#define STATS_H_


//root and gsl librarys


//#include <ReadWriteNet.h>
#include <ReadWrite.h>
#include "DataMine.h"
#include <experimental/type_traits>
#include <experimental/optional>
//#include <type_traits>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/skewness.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/math/statistics/t_test.hpp>

/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/plot.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cvplot.h"
*/

#include <algorithm>
#include <numeric>
#include <thread>
#include <random>




#include <typeinfo>

#define GRETL 1

#if GRETL == 1

#include <gretl/libgretl.h>//this header causes name crashing with #include <caffe2/core/init.h> and #include <caffe2/core/tensor.h> these in Ensemble.h
#include <gretl/describe.h>
//#include <gretl/boxplots.h>

extern "C" {
    #include "gretl/boxplots.h"
}
//#include <gretl/gretl_list.h>//this some how cause a problem with bp::child c("/usr/bin/unzip",args, bp::std_out > sstream); in ReadWrite void Unzip
								//if I put a break point in here variables_of_interest_vect[count] = variables_of_interest; in MBLStats and hold a second then
								//the problem goes away
#include <gretl/graphing.h>
#endif

namespace bacc = boost::accumulators;
typedef boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::density> > acc;
typedef boost::iterator_range<std::vector<std::pair<double, double> >::iterator > histogram_type;

//typedef boost::accumulators::accumulator_set<double, bacc::features<bacc::tag::density> > facc;
typedef bacc::accumulator_set< double, bacc::features<
			bacc::tag::min,
			bacc::tag::max,
			bacc::tag::mean,
			bacc::tag::median,
			bacc::tag::variance,
			bacc::tag::skewness,
			bacc::tag::density > > stat_acc;//w_acc;





namespace Stats{


struct Histogram {

	int count = 0;//number of observations in the data set-> or ncases
	std::vector<short int> bin_index;//hold the bin the values(cases) in data ex: data[0] = -5 whic in at the histogram
					//is bin 3 then bin_index[0] = 3
	//std::vector<double> marginals;
	std::vector<float> marginals;
	std::vector<double> bins;
	std::vector<float > hist_vect;//marginal-> or percent of data that falls in that bin
	std::vector<float > hist_vect_bin;//bin values so (-inf,1],(1,2],....(n,inf) were 1,2,...,n represent the range of the bins
	std::vector<std::vector<int > > contingency_table;


};


template<typename T>
std::map<std::string, double > CalcSimpleStats(std::vector<T> &data,Histogram &hist_st, int n_bins = 10, bool show_stats = false);

void PlotHist(Histogram &hist_st, std::string file_path, std::string file_name);
template< typename T >
void PlotLine(std::vector<T> &data, std::string name);
template< typename T >
void ComputeHistograms(std::vector<T> &data,Histogram &hist_st, int n_bins = 10);
template<typename T, typename T1>
void LaggedMI(std::vector<T> pred_vars, std::vector<T1 > target, int n_bins, int min_lag, int max_lag, int lag_step);
void LaggedMI(std::vector<std::vector<double> > embedVect, std::vector<double > vect_y_alligned, int n_bins=10);
double DiscreteMI (Histogram &pred_hist_st, Histogram &target_hist_st);
double Entropy (Histogram &hist_st);
void ACF(std::vector<std::vector<double> > &embedVect, int lag = 1);
template<typename T, typename T1>
double correlation(int var_num, std::vector<std::vector<T> > &data, std::vector<T1> &target);
double ComputeV (Histogram &pred_hist_st, Histogram &target_hist_st);
void two_samples_t_test_equal_sd(double Sm1,double Sd1,unsigned Sn1,double Sm2,double Sd2,unsigned Sn2,double alpha);
void two_samples_t_test_unequal_sd(double Sm1,double Sd1,unsigned Sn1,double Sm2,double Sd2,unsigned Sn2,double alpha);

template<typename T>
void CountNANs(std::vector<T> &data, std::vector<int> &nan_location);

template<typename T>
void CountNANs(std::vector<std::vector<T> > &data,std::vector<int> &nan_location);

template<typename T>
void GetUniqueCounts(std::vector<T> &data, std::map<T, int> &value_count, bool skip_nan = true, bool cumulative = false);

template<typename T>
void GetUniqueCounts(std::vector<std::vector<T> > &data, std::initializer_list<int> columns_to_get, std::map<std::string, int> &value_count, bool skip_nan = true, bool cumulative = false);


template<typename T, typename T1, typename T2>
std::vector<double> LombScargle(std::vector<T> &times, const std::vector<T1> &values, std::vector<T2> &frequencies, bool center_data=true , std::string normalization= "standard");

template<typename T, typename T1>
std::vector<T> AutoFrequency(std::vector<T1> times, std::experimental::optional<double> samples_per_peak=5, std::experimental::optional<double> nyquist_factor=5, std::experimental::optional<double> minimum_frequency=0.0,
		std::experimental::optional<double> maximum_frequency=0.00, std::experimental::optional<bool> return_freq_limits=false);
/*
 *
 * search and drop correlated variables
 *	void DROP_CORRELATED_VARIABLES()
 *
 *
 *
 */


#if GRETL == 1
void SetGretlPaths(std::string path);
void ShapiroWilksGretl(std::map<std::string, std::vector<double> > &data, bool transform=false);
double HurstExponentGretl(const int *list, DATASET *dset, gretlopt opt,PRN *prn, bool no_plot = false);
int ArmaEstimateGretl (DATASET *dset, PRN *prn, MODEL *model, int p, int d, int q, int position_of_dependent_variable, int *use_residuals = 0);
std::pair<std::map<double, int>,std::map<double, int> > CorrgramGretl(int varno, int max_lag, DATASET *dset, gretlopt opt, PRN *prn);
std::map<double, double> PeriodogramGretl(int varno, int max_lag, DATASET *dset, gretlopt opt, PRN *prn, bool no_plot = false);
Summary *SummaryGretl(const DATASET *dset, PRN *prn, int *err);

#endif


void AnalyzeACFPACF(std::pair<std::map<double, int>,std::map<double, int> > acf_pacf_values, int size_data);


}


#endif /* STATS_H_ */
