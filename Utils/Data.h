/*
 * Data.h
 *
 *  Created on: Mar 18, 2019
 *      Author: ryan
 */

#pragma once
#ifndef DATA_H_
#define DATA_H_

//#include <ReadWriteNet.h>
#include <ReadWrite.h>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <typeinfo>
#include <thread>
#include <cxxabi.h>
#include <unordered_map>
#include <boost/lexical_cast.hpp>
#define DATA_MAP 1

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
	 * T, T1, and Tm will be the types that you are will be feeding into the model
	 *
	 * @tparam T: The features data,typically vector<vector< float> >
	 * @tparam T1: The labels data, vector<vector< float> > for classification and vector< float> for regression
	 * @tparam Tg: Tg is just a data type of a vector. were the vector is the data that you want to split into train, test, validate
	 * 				default is a string
	 * @tparam Tm: The labels mask, vector<vector< float> >
	 */
	template<typename T, typename T1, typename Tg = std::string, typename Tm = std::vector<std::vector<int> > >
	struct SplitData{

		enum split_or_not{
			split_data_only,
			split_and_all_data,
			no_split
		};

		bool is_map =true;
		T train_features_;
		T1 train_labels_;
		T1 train_labels_one_hot_;
		Tm train_labels_mask_;

		T test_features_;
		T1 test_labels_;
		T1 test_labels_one_hot_;
		Tm test_labels_mask_;

		T validate_features_;
		T1 validate_labels_;
		T1 validate_labels_one_hot_;
		Tm validate_labels_mask_;


		std::vector<Tg> to_get;//the data that they want to get

		std::vector<boost::posix_time::ptime> DateToIter;
		std::multimap<boost::posix_time::ptime, std::vector<long double>> NonAdjustedSprdData;
		std::vector<std::vector<float > > all_data_features_;
		std::vector<float> all_data_labels_;
		std::vector<std::vector<float> > all_data_labels_hot_;//For 1 Hot encoding

		template<typename T2, typename T3>
		void FnlDataToStruct(T2 &data, T3 &labels, int TRTEVA,  split_or_not sp = split_data_only);

		template<typename T2, typename T3, typename T4, typename T5>
		void BuildIters(T2 &data, T3 &labels, T4 &id_labels_to_labels, T5 &masks, int TRTEVA, split_or_not split);

		template<typename T2, typename T3>
		void FnlDataToStructVector(T2 &data, T3 &labels, int TRTEVA);

		template<typename T2, typename T3>
		void FnlDataToStructMap(T2 &data, T3 &labels, int TRTEVA);
	};



	std::vector<float> VectorVectorStringToVectorFloat(std::vector<std::vector<std::string> > &data_in, int column);


	template< typename T, typename T1, typename T2, typename T3 >
	void GroupBy(std::set<T> &set_2_convert,std::vector<std::vector<T1> > &data, T2 &grouped, int group_by_column, int set_column, T3 &var_columns);
	template< typename T, typename T1, typename T2, typename T3 >
	void GroupByAllNumeric(std::set<T> &set_2_convert,std::vector<std::vector<T1> > &data, T2 &grouped, int group_by_column, int set_column, T3 &var_columns);

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
	void NonAdjustedData(std::string fileName, std::multimap<boost::posix_time::ptime, std::vector<long double>> &Sprd);
	void SprdData(std::string fileName, std::multimap<boost::posix_time::ptime, std::vector<long double>> &Sprd, bool ud=false, bool include_change=false);
	void InputVar(std::string fileName, std::multimap<boost::posix_time::ptime, std::vector<long double>> &InpuVar,long double multiplier=1.00);

	// the below use to be void Embed(..........
		void EmbedNoReturn(std::multimap<boost::posix_time::ptime, std::vector<long double>> &data ,int m, int d, bool skip = false);

	void Print(std::multimap<boost::posix_time::ptime, std::vector<long double>> &data);


	//template<typename T, typename T1>
	struct ModelConfig{
		std::string model_name = "what_what";//used as we are incrementing trying to get the best
		float no_train_thresh = .55;//% accurcacy we would like to be > for non trained data
		float all_thresh = .55;//% accurcacy we would like to be > for all the data
		std::string *model_name_ptr = &model_name;

		inline std::string init_model_name(){return "init"+this->model_name;}
		std::string final_model_name = "what_final";//the best model achieved for this configuration
		//what about multiple model paths, for different neural networks and xgboost
		std::string model_path = "/home/ryan/workspace/ModelAnalizeCompareC2/";
		std::string data_path = "/home/ryan/workspace/ModelAnalizeCompareC2/";
		int TRTEVA = 0;
		std::string feature_vars;
		std::string spread_filename ="Sprdfilename.csv";
		std::string activation;
		int n_classes =2;
		int vc = 0;
		int n_features = 0;
		std::string chart_path = data_path+"PnLCharts/";
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
