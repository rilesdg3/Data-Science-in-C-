/*
 * SaveData.h
 *
 *  Created on: May 27, 2020
 *      Author: ryan
 */


#ifndef SAVEDATA_H_
#define SAVEDATA_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <zip.h>
#include <dirent.h>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>
#include <tuple>
#include <regex>
#include <chrono>
#include "boost/date_time/posix_time/posix_time.hpp" //include all types plus i/o
#include <boost/date_time/parse_format_base.hpp>
#include "boost/date_time/gregorian/gregorian.hpp"
#include <set>
#include <DataFrame/DataFrame.h>
#include <boost/any.hpp>


#include "boost/process.hpp"
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/iostreams/seek.hpp>

namespace ReadWrite{

//namespace bp = ::boost::process;


template<typename T>
bool IsNumber(T x);
typedef hmdf::StdDataFrame<ulong> MyDataFrame;

template<typename T>
void GetDataTypes(std::string file_name, std::vector<std::vector<std::string> > &names_types,int &n_rows, int &n_columns, bool header=false);


/*
 * what dat afuck
 *
 */
template<typename T>
void ReadCSV(std::string file_name, MyDataFrame &df, std::vector<std::string> &column_names_string, bool header = false);
template<typename T>
void ReadCSV(std::string file_name, std::vector<T> &df, std::vector<std::string> &column_names_string, bool header = false);



template<typename T1, typename T2>
void ReadCSV(std::string file_name, std::map<T1, std::vector<T2 > > &the_data, bool header, std::set<int> &skip_columns);
void ReadFile(std::string file_name, std::string &data);
void Unzip(std::string file_path_name, std::string &data);
void ReadCSVZIP(std::string file_name, std::string &data);



//void SaveData(std::string file_path, std::string name, const std::map<std::string, std::map<boost::gregorian::date, std::vector<long double> > > &data);
template<typename T, typename T1>
void SaveData(std::string file_path, std::string name, const T1 &data);
template<typename T, typename T1 >
void SaveDataVector(std::string file_path, std::string name, const T1 &data);

template<typename T, typename T1>
void ParseString(std::string &data, std::multimap<T, std::vector<std::vector<T1> > > &final_data, int column_idx_to_key,
		int outer_vect_size, int inner_vect_size, std::set<int> skip_columns);
template<typename T1>
void ParseString(std::string &data, std::multimap<boost::posix_time::ptime, std::vector<std::vector<T1> > > &final_data, int column_idx_to_key,
		int outer_vect_size, int inner_vect_size, std::set<int> skip_columns);

std::string DateToString(boost::posix_time::ptime datetime);
std::vector<std::string> iterdir(std::string file_path, std::set<std::string> &find_in_name);
std::vector<std::string> iterdir(std::string file_path, std::string &find_in_name);
std::vector<std::string> iterdir(std::string file_path, std::set<std::string> &find_in_name, std::string to_append);


boost::gregorian::date DateFromString(std::string time);

template<typename T>
boost::posix_time::ptime EpochToPtimeNanoSeconds(const T &epoch);

std::string DateTimeToString(boost::posix_time::ptime datetime);


}

#endif /* SAVEDATA_H_ */
