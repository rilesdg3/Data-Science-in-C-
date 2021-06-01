/*
 * Stats.h
 *
 *  Created on: Mar 18, 2019
 *      Author: ryan
 */
#pragma once
#ifndef STATS_H_
#define STATS_H_


//root and gsl librarys


//#include <ReadWriteNet.h>
#include <ReadWrite.h>
#include "DataMine.h"
#include <experimental/type_traits>
#include <experimental/optional>
//#include <type_traits>
//#include <minpack.h>
#include <cminpack.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/weighted_mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/skewness.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/rolling_window.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_variance.hpp>

#include <boost/math/statistics/t_test.hpp>

/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cvplot.h"
*/

#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <algorithm>
#include <numeric>
#include <thread>
#include <random>
#include <math.h>
#include <typeinfo>
#include <assert.h>
#include <gnuplot-iostream.h>
//#define real __cminpack_real__

#define GRETL 1

#if GRETL == 1

#include <gretl/libgretl.h>//this header causes name crashing with #include <caffe2/core/init.h> and #include <caffe2/core/tensor.h> these in Ensemble.h
#include <gretl/describe.h>
//#include <gretl/boxplots.h>

extern "C" {
    #include "gretl/boxplots.h"

//#include "gretl/genparse.h"
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


enum {
    F1_MIN = 1 << 8,
    F_ABS,
    F_SGN,
    F_CEIL,
    F_FLOOR,
    F_SIN,
    F_COS,
    F_TAN,
    F_ASIN,
    F_ACOS,
    F_ATAN,
    F_SINH,
    F_COSH,
    F_TANH,
    F_ASINH,
    F_ACOSH,
    F_ATANH,
    F_LOG,
    F_LOG10,
    F_LOG2,
    F_EXP,
    F_SQRT,
    F_GAMMA,
    F_LNGAMMA,
    F_DIGAMMA,
    F_TRIGAMMA,
    F_INVMILLS,
    F_ROUND,
    F_CNORM,
    F_DNORM,
    F_QNORM,
    F_CARG,
    F_CMOD,
    F_REAL,
    F_IMAG,
    F_LOGISTIC,
    FP_MAX,      /* separator: end of pointerized functions */
    F_CONJ,
    F_TOINT,
    F_DIFF,	  /* first difference */
    F_LDIFF,	  /* log difference */
    F_SDIFF,	  /* seasonal difference */
    F_SORT,	  /* ascending sort */
    F_DSORT,	  /* descending sort */
    F_RANKING,
    F_ODEV,	  /* orthogonal deviation */
    F_NOBS,
    F_CUM,
    F_MISSING,
    F_DATAOK,
    F_MISSZERO,
    F_ZEROMISS,
    F_MEDIAN,
    F_GINI,
    F_SUMALL,
    F_MIN,
    F_MAX,
    F_SKEWNESS,
    F_KURTOSIS,
    F_SST,
    F_SUMR,
    F_SUMC,
    F_PRODR,
    F_PRODC,
    F_MEANR,
    F_MEANC,
    F_CHOL,
    F_INV,
    F_INVPD,
    F_GINV,
    F_DIAG,
    F_TRANSP,
    F_VEC,
    F_VECH,
    F_UNVECH,
    F_ROWS,
    F_COLS,
    F_DET,
    F_LDET,
    F_TRACE,
    F_NORM1,
    F_INFNORM,
    F_RCOND,
    F_RANK,
    F_OBSNUM,
    F_ISDISCR,
    F_ISDUMMY,
    F_TYPEOF,
    F_EXISTS,
    F_NELEM,
    F_PDF,
    F_PVAL,
    F_CDF,
    F_INVCDF,
    F_CRIT,
    F_URCPVAL,
    F_RANDGEN,
    F_MRANDGEN,
    F_RANDGEN1,
    F_VALUES,
    F_UNIQ,
    F_NULLSPC,
    F_MEXP,
    F_MINC,
    F_MAXC,
    F_MINR,
    F_MAXR,
    F_IMINC,
    F_IMAXC,
    F_IMINR,
    F_IMAXR,
    F_FFT,
    F_FFT2,
    F_FFTI,
    F_UPPER,
    F_LOWER,
    F_POLROOTS,
    F_OBSLABEL,
    F_BACKTICK,
    F_STRLEN,
    F_VARNAME,
    F_VARNAMES,
    F_VARNUM,
    F_TOLOWER,
    F_TOUPPER,
    F_IRR,
    F_ERRMSG,
    F_GETENV,
    F_NGETENV,
    F_PEXPAND,
    F_FREQ,
    F_ISNAN,
    F_TYPESTR,
    F_STRSTRIP,
    F_REMOVE,
    F_ATOF,
    F_MPI_RECV,
    F_EASTER,
    F_CURL,
    F_NLINES,
    F_ARRAY,
    F_TRAMOLIN,
    F_CNUMBER,
    F_ECDF,
    F_SLEEP,
    F_GETINFO,
    F_CDUMIFY,
    F_GETKEYS,
    F_MCORR,
    F_ISCMPLX,
    F_CTRANS,
    F_MLOG,
    F_BARRIER,
    HF_JBTERMS,
    F1_MAX,	  /* SEPARATOR: end of single-arg functions */
    HF_LISTINFO,
    F_SUM,
    F_MEAN,
    F_VCE,
    F_SD,
    F_ARGNAME,
    F_T1,
    F_T2,
    F_COV,
    F_SDC,
    F_CDEMEAN,
    F_MCOV,
    F_DUMIFY,
    F_SORTBY,
    F_RUNIFORM,
    F_RNORMAL,
    F_FRACDIFF,
    F_BOXCOX,
    F_ZEROS,
    F_ONES,
    F_MUNIF,
    F_MNORM,
    F_QFORM,
    F_QR,
    F_EIGSYM,
    F_QUANTILE,
    F_CMULT,	  /* complex multiplication */
    F_HDPROD,     /* horizontal direct product */
    F_CDIV,	  /* complex division */
    F_MXTAB,
    F_MRSEL,
    F_MCSEL,
    F_STRSTR,
    F_INSTRING,
    F_CNAMESET,
    F_RNAMESET,
    F_LJUNGBOX,
    F_MSORTBY,
    F_MSPLITBY,
    F_LINCOMB,
    F_IMHOF,
    F_XMIN,
    F_XMAX,
    F_FRACLAG,
    F_MREV,
    F_DESEAS,
    F_PERGM,
    F_NPV,
    F_DSUM,
    F_POLYFIT,
    F_INLIST,
    F_ISCONST,
    F_INBUNDLE,
    F_CNAMEGET,
    F_RNAMEGET,
    F_PNOBS,
    F_PMIN,
    F_PMAX,
    F_PSUM,
    F_PMEAN,
    F_PXSUM,
    F_PXNOBS,
    F_PSD,
    F_PSHRINK,
    F_RANDINT,
    F_MREAD,
    F_BREAD,
    F_GETLINE,
    F_ISODATE,
    F_JULDATE,
    F_READFILE,
    F_PRINTF,
    F_SPRINTF,
    F_MPI_SEND,
    F_BCAST,
    F_ALLREDUCE,
    F_GENSERIES,
    F_KPSSCRIT,
    F_STRINGIFY,
    F_SQUARE,
    F_SEASONALS,
    F_DROPCOLL,
    F_KSIMDATA,
    F_HFDIFF,
    F_HFLDIFF,
    F_NAALEN,
    F_KMEIER,
    F_NORMTEST,
    F_COR,
    F_LRCOVAR,
    F_JSONGETB,
    F_FIXNAME,
    F_ATAN2,
    F_CCODE,
    F_LSOLVE,
    F_STRFTIME,
    F_STRPTIME,
    F_CONV2D,
    F_FLATTEN,
    F_IMAT,
    F_COMPLEX,
    F_RANDPERM,
    F_STDIZE,
    F_CSWITCH,
    F_PSDROOT,
    F_INSTRINGS,
    F_STRVALS,
    F_FUNCERR, /* legacy */
    F_ERRORIF,
    F_BINCOEFF,
    F_ASSERT,
    F2_MAX,	  /* SEPARATOR: end of two-arg functions */
    F_WMEAN,
    F_WVAR,
    F_WSD,
    F_LLAG,
    F_HFLAG,
    F_PRINCOMP,
    F_BFGSMAX,
    F_MSHAPE,
    F_SVD,
    F_TRIMR,
    F_TOEPSOLV,
    F_CORRGM,
    F_SEQ,
    F_REPLACE,
    F_STRNCMP,
    F_BESSEL,
    F_WEEKDAY,
    F_MONTHLEN,
    F_EPOCHDAY,
    F_KDENSITY,
    F_SETNOTE,
    F_BWFILT,
    F_VARSIMUL,
    F_STRSUB,
    F_REGSUB,
    F_MLAG,
    F_EIGSOLVE,
    F_SIMANN,
    F_HALTON,
    F_MWRITE,
    F_BWRITE,
    F_AGGRBY,
    F_IWISHART,
    F_SSCANF,
    F_SUBSTR,
    F_REDUCE,
    F_SCATTER,
    F_MWEIGHTS,
    F_MGRADIENT,
    F_MLINCOMB,
    F_HFLIST,
    F_NMMAX,
    F_GSSMAX,
    F_NPCORR,
    F_DAYSPAN,
    F_SMPLSPAN,
    F_FDJAC,
    F_NUMHESS,
    F_STRSPLIT,
    F_HPFILT,
    F_XMLGET,
    F_JSONGET,
    F_FEVD,
    F_LRVAR,
    F_BRENAME,
    F_ISOWEEK,
    F_BKW,
    F_FZERO,
    F_EIGGEN,
    F_EIGEN,
    F_SCHUR,
    F_RESAMPLE,
    F_STACK,
    F_GEOPLOT,
    F_VMA,
    F_FCSTATS,
    F_BCHECK,
    HF_REGLS,
    F3_MAX,       /* SEPARATOR: end of three-arg functions */
    F_BKFILT,
    F_MOLS,
    F_MPOLS,
    F_MRLS,
    F_FILTER,
    F_MCOVG,
    F_KFILTER,
    F_KSMOOTH,
    F_KDSMOOTH,
    F_KSIMUL,
    F_NRMAX,
    F_LOESS,
    F_GHK,
    F_QUADTAB,
    F_ISOCONV,
    F_QLRPVAL,
    F_BOOTCI,
    F_BOOTPVAL,
    F_MOVAVG,
    F_DEFARRAY,
    F_DEFBUNDLE,
    F_DEFLIST,
    F_DEFARGS,
    F_KSETUP,
    F_BFGSCMAX,
    F_SVM,
    F_IRF,
    F_NADARWAT,
    F_FEVAL,
    F_CHOWLIN,
    F_TDISAGG,
    F_HYP2F1,
    F_MIDASMULT,
    HF_CLOGFI,
    FN_MAX,	  /* SEPARATOR: end of n-arg functions */
};


namespace Stats{

typedef hmdf::StdDataFrame<ulong> MyDataFrame;

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
std::vector<double> LaggedMI(std::vector<T> pred_vars, std::vector<T1 > target, int n_bins, int min_lag, int max_lag, int lag_step);
void LaggedMI(std::vector<std::vector<double> > embedVect, std::vector<double > vect_y_alligned, int n_bins=10);
double DiscreteMI (Histogram &pred_hist_st, Histogram &target_hist_st);
double Entropy (Histogram &hist_st);
void ACF(std::vector<std::vector<double> > &embedVect, int lag = 1);
template<typename T, typename T1>
double correlation(int var_num, std::vector<std::vector<T> > &data, std::vector<T1> &target);
template<typename T, typename T1>
double correlation(std::vector<T> &data, std::vector<T1> &target);
double ComputeV (Histogram &pred_hist_st, Histogram &target_hist_st);
std::vector<std::vector<int> > Combinations(std::vector<int> iterable, int r);
template<typename T>
std::vector<double> DiffSeries(std::vector<T> &data);
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
std::vector<T> AutoFrequency(std::vector<T1> &times, std::experimental::optional<double> samples_per_peak=5, std::experimental::optional<double> nyquist_factor=5, std::experimental::optional<double> minimum_frequency=0.0,
		std::experimental::optional<double> maximum_frequency=0.00, std::experimental::optional<bool> return_freq_limits=false);

template<typename T, typename T1>
std::map<double, double> false_alarm_level(std::vector<T> &times, std::vector<T1> &values, std::vector<double> &false_alarm_probability, std::experimental::optional<std::string> method="baluev",std::experimental::optional<double> samples_per_peak=5,
		std::experimental::optional<double> nyquist_factor=5, std::experimental::optional<double> minimum_frequency=0.0,std::experimental::optional<double> maximum_frequency=0.0);

template<typename T, typename T1>
double inv_fap_baluev(double fap, double maximum_frequency, std::vector<T> &times, std::vector<T1> &values, std::string normalization="standard");

template<typename T, typename T1>
double inv_fap_naive(double fap, double fmax, std::vector<T> &t, std::string normalization="standard");

double inv_fap_single(double fap, int N, std::string normalization, int dH=1, int dK=3);

template<typename T, typename T1>
double fap_baluev(double Z, double fmax, std::vector<T> &t, std::vector<T1> &y, double dy =1 ,std::string normalization="standard");


double fap_single(double z, int N, std::string normalization, int dH=1, int dK=3);

template<typename T, typename T1>
double tau_davies(double Z, double fmax, std::vector<T> &t, std::vector<T1> &y, double dy = 1.000, std::string normalization="standard", double dH=1, double dK=3);

/*
 * For FAL
 * false_alarm_level->_core.py
 * 		false_alarm_level->_statistics.py
 * inv_fap_baluev->_statistics.py
 * inv_fap_naive->_statistics.py
 * inv_fap_single->_statistics.py
 * fap_baluev->_statistics.py
 * 		fap_single->_statistics.py
 * 		tau_davies->_statistics.py
 * optimize.root->root.py
 * root->_root
 * root_leastsq->_root.py
 * leastlsq->minpack.py
 *
*/





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
void ShapiroWilksGretl(DATASET *dset, std::vector<double> variables, bool transform=false);
double HurstExponentGretl(const int *list, DATASET *dset, gretlopt opt,PRN *prn, bool no_plot = false, std::string path_name="-1");
int ArmaEstimateGretl (DATASET *dset, PRN *prn, MODEL *model, int p, int d, int q, int position_of_dependent_variable, int *use_residuals = 0);
std::pair<std::map<double, int>,std::map<double, int> > CorrgramGretl(int varno, int max_lag, DATASET *dset, gretlopt opt, PRN *prn,  std::string path_name="-1");
std::map<double, double> PeriodogramGretl(int varno, int max_lag, DATASET *dset, gretlopt opt, PRN *prn, bool no_plot = false,  std::string path_name="-1");
Summary *SummaryGretl(const DATASET *dset, PRN *prn, int *err);

#endif


void AnalyzeACFPACF(std::pair<std::map<double, int>,std::map<double, int> > acf_pacf_values, int size_data);


}


#endif /* STATS_H_ */
