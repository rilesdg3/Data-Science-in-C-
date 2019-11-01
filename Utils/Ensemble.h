/*
 * Ensemble.h
 *
 *  Created on: Aug 9, 2019
 *      Author: ryan
 */


#pragma once

#ifndef ENSEMBLE_H_
#define ENSEMBLE_H_



#include "ReadWriteNet.h"
#include "blob.h"
#include "model.h"
#include "Data.h"

#include <caffe2/core/init.h>
#include <caffe2/core/tensor.h>

#include <limits>
#include <numeric>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/stats.hpp>

#include <cvplot.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/plot.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace bacc = boost::accumulators;



static struct Data{
	std::vector<boost::posix_time::ptime> DateToIter;
	std::multimap<boost::posix_time::ptime, std::vector<long double>> NonAdjustedSprdData;
	std::vector<std::vector<float > > Features;
	std::vector<int> Labels;
	std::vector<std::vector<int> >LabelsHot;//For 1 Hot encoding
}TrainData, TestData, ValidateData, AllData;

static struct Results{
	float best = 0.0;
	float allBuys = 0.0;
	float bestNonTrained = 0.0;
	float validateResults = 0.0;
	float testResults = 0.0;
	float allResults = 0.0;
	float combinedResults =0.0;
	float diffResults = 0.0;
	bool betterThan=false;


}Results;


typedef bacc::accumulator_set< double, bacc::stats<
			bacc::tag::min,
			bacc::tag::max,
			bacc::tag::mean,
			bacc::tag::median,
			bacc::tag::variance > > w_acc;



template<typename T, typename T1>
struct Ensemble{

	std::vector<T> all_pred_vect_tmp;//this holds the predictions from each individual model
	std::vector<T1> all_pred_fnl;//final preditions for an ensemble, the outer vector reprsents what we
									//are pedicting for and the inner vector represents the predictions
									//from each individual model


	//std::vector<std::multimap<boost::posix_time::ptime, float > > all_pred_vect_tmp;//this holds the predictions from each individual model
	//std::multimap<boost::posix_time::ptime, std::vector< float > > all_pred_fnl;//final preditions for an ensemble

	string model_name = "what_what";//used as we are incrementing trying to get the best
	float no_train_thresh = .55;//% accurcacy we would like to be > for non trained data
	float all_thresh = .55;//% accurcacy we would like to be > for all the data
	//string *model_name_ptr = &model_name;

	inline string init_model_name(){return "init"+this->model_name;}
	string final_model_name = "what_final";//the best model achieved for this configuration
	//what about multiple model paths, for different neural networks and xgboost
	string model_path = "/home/ryan/workspace/ModelAnalizeCompareC2/";
	string data_path = "/home/ryan/workspace/ModelAnalizeCompareC2/";
	int TRTEVA = 0;
	string feature_vars;
	string spread_filename ="Sprdfilename.csv";
	string activation;
	int n_classes =1;
	int vc = 0;
	int n_features = 0;
	string chart_path = data_path+"PnLCharts/";
	std::vector<int> n_hide;
	/* this is to be deleted when n_hide vector is implemented
					int n_hide = 0;
					int n_hide1 = 0;
					int n_hide2 = 0;
					int n_hide3 = 0;*/

	float learning_rate = 0.0;


};



//typedef std::tuple<float,int,int,int,int,boost::posix_time::ptime,boost::posix_time::ptime,int,int,int,boost::posix_time::ptime,std::vector<long double>> StatsTuple;
typedef std::tuple<float,long double,long double,long double,long double,boost::posix_time::ptime,boost::posix_time::ptime,long double,long double,long double,
		boost::posix_time::ptime,std::vector<long double>, std::vector<long double>  > StatsTuple;

//void Embed(std::multimap<boost::posix_time::ptime, std::vector<long double>> &data ,int m, int d, bool skip=false);
std::multimap<boost::posix_time::ptime, std::vector<long double>> BuildFnlData(std::vector<std::multimap<boost::posix_time::ptime, std::vector<long double> > > &what);

std::vector<std::vector<long double> > BuildFnlDataVect(std::vector<std::multimap<boost::posix_time::ptime, std::vector<long double> > > &what);
void FnlDataToStruct(std::multimap<boost::posix_time::ptime, std::vector<long double> > &data, int TRTEVA);



void PotentialEnesmbleCheck(int stat, std::map<int, std::map<vector<int>, StatsTuple > > &potential_ensembles, vector<int> ensemble, StatsTuple ensemble_stats);
void FindIntersectingEnsemblesSets(std::map<int, std::map<vector<int>, StatsTuple > > &ensemble, std::map<int, std::map<vector<int>, StatsTuple > > &ensemble1,
		std::map<int, std::map<vector<int>, StatsTuple > > &ensemble2);
void FindIntersectingEnsemblesMap(std::map<int, std::map<vector<int>, StatsTuple > > &ensemble, std::map<int, std::map<vector<int>, StatsTuple > > &ensemble1,
		std::map<int, std::map<vector<int>, StatsTuple > > &ensemble2);

void PlotAndSavePnL(vector<int> &models_in_ensemble, std::map<vector<int>, StatsTuple > &ensemble);
void CreateSets(int n_TRTEVA = 9, int n_TETRVA =18, int n_VATRTE = 27);
bool AllowCombo(vector<int> combos, int n_models_per_set);
vector<vector<int> > Combinations(vector<int> iterable, int r);
template <typename T>
void BuildAllPredVectFnl(T &ensemble);

template<typename T, typename T1>
std::map<int, std::map<vector<int>, StatsTuple > > AnalizeEnsembles(vector<vector<int> > &combos, T &ensemble, T1 &labels,bool past_year = false);

std::map<int, std::map<vector<int>, StatsTuple > > AnalizeEnsemblesPnL(vector<vector<int> > &combos, std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long> > &ensemble,
		string spread_filename, bool past_year = false);

template<typename T>
void PredictEnsemble(T &all_pred_fnl);

template<typename T>
void BuildEnsemble(T &all_pred_fnl);

template <typename T, typename T1>
void BuildEnsembleThreading(T &all_pred_fnl, T1 &labels);

template <typename T, typename T1>
void GoThroughCombos(int start, int end, vector<vector<int> > &combos, T &all_pred_fnl, T1 &labels);

template <typename T>
void GoThroughCombosBinary(int start, int end, vector<vector<int> > &combos);



namespace CPCO {
void ModelNames();

}




template<typename T,typename T1,typename T2>
void PredictAll(T &ens, T1 &features, T2 &labels );
template<typename T>
void PredictDay(caffe2::NetDef &init_model, caffe2::NetDef &predict_model, string model_name,T &pred_vect);
void PredictNonTrained(caffe2::NetDef &init_model, caffe2::NetDef &predict_model);

template<typename T>
void TestModel(T &ens);





#endif /* ENSEMBLE_H_ */
