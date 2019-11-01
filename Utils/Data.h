/*
 * Data.h
 *
 *  Created on: Mar 18, 2019
 *      Author: ryan
 */

#pragma once
#ifndef DATA_H_
#define DATA_H_

#include <ReadWriteNet.h>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <typeinfo>
#include <thread>
#include <cxxabi.h>


namespace MyData {


	struct classcomp {
	  bool operator() (const double& lhs, const double& rhs) const
	  {return lhs>rhs;}
	};

	// class generator:
	struct c_unique {
	  int current;
	  c_unique() {current=0;}
	  int operator()() {return ++current;}
	};// UniqueNumber;


	/*
	 * @brief: holds the data for train, validation, test and provides functions for
	 * dividing the spliting the data into train, validate, test
	 *
	 * @tparam T: The features data
	 * @tparam T1: The labels data
	 */
	template<typename T, typename T1>
	struct SplitData{

		enum split_or_not{
			split_data_only,
			split_and_all_data,
			no_split
		};

		bool is_map =true;
		T train_features_;
		T1 train_labels_;

		T test_features_;
		T1 test_labels_;

		T validate_features_;
		T1 validate_labels_;

		std::vector<boost::posix_time::ptime> DateToIter;
		std::multimap<boost::posix_time::ptime, std::vector<long double>> NonAdjustedSprdData;
		std::vector<std::vector<float > > all_data_features_;
		std::vector<float> all_data_labels_;
		std::vector<std::vector<float> > all_data_labels_hot_;//For 1 Hot encoding

		template<typename T2, typename T3>
		void FnlDataToStruct(T2 &data, T3 &labels, int TRTEVA,  split_or_not sp = split_data_only);

		template<typename T2, typename T3>
		void FnlDataToStructVector(T2 &data, T3 &labels, int TRTEVA);

		template<typename T2, typename T3>
		void FnlDataToStructMap(T2 &data, T3 &labels, int TRTEVA);
	};



	std::vector<float> VectorVectorStringToVectorFloat(std::vector<std::vector<string> > &data_in, int column);


	template< typename T, typename T1, typename T2, typename T3 >
	void GroupBy(std::set<T> &set_2_convert,std::vector<std::vector<T1> > &data, T2 &grouped, int group_by_column, int set_column, T3 &var_columns);

	template<typename T, typename T1 >
	void GroupBy(T &data, T1 &grouped, int group_by_column, int value_column);

	void EmbedVect(std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd,
	std::vector<std::vector<double> > &vect, int vect_data_it, int m, int d, bool skip);
	std::vector<std::vector<double> > EmbedThreading(std::multimap<double, std::vector<float>, classcomp >  &data ,int m, int d, bool skip,
	std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd, bool pnt_data = false);
	std::vector<std::vector<double> > Embed(std::multimap<double, std::vector<float>, classcomp >  &data ,int m, int d, bool skip);

	std::vector<double> AlignedYvectWithEmbedX(std::multimap<double, std::vector<float>, classcomp >  &data,
	std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd);


	inline int ZeroOne(long double x);
	inline int ZeroOneNegOne(long double x);
	void NonAdjustedData(string fileName, std::multimap<boost::posix_time::ptime, std::vector<long double>> &Sprd);
	void SprdData(string fileName, std::multimap<boost::posix_time::ptime, std::vector<long double>> &Sprd, bool ud=false, bool include_change=false);
	void InputVar(string fileName, std::multimap<boost::posix_time::ptime, std::vector<long double>> &InpuVar,long double multiplier=1.00);

	// the below use to be void Embed(..........
		void EmbedNoReturn(std::multimap<boost::posix_time::ptime, std::vector<long double>> &data ,int m, int d, bool skip = false);

	void Print(std::multimap<boost::posix_time::ptime, std::vector<long double>> &data);


	//template<typename T, typename T1>
	struct ModelConfig{
		string model_name = "what_what";//used as we are incrementing trying to get the best
		float no_train_thresh = .55;//% accurcacy we would like to be > for non trained data
		float all_thresh = .55;//% accurcacy we would like to be > for all the data
			string *model_name_ptr = &model_name;

		inline string init_model_name(){return "init"+this->model_name;}
		string final_model_name = "what_final";//the best model achieved for this configuration
		//what about multiple model paths, for different neural networks and xgboost
		string model_path = "/home/ryan/workspace/ModelAnalizeCompareC2/";
		string data_path = "/home/ryan/workspace/ModelAnalizeCompareC2/";
		int TRTEVA = 0;
		string feature_vars;
		string spread_filename ="Sprdfilename.csv";
		string activation;
		int n_classes =2;
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



};

#endif /* DATA_H_ */
