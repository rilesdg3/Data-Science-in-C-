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

namespace ReadWrite{


template<typename T>
bool IsNumber(T x);
typedef hmdf::StdDataFrame<ulong> MyDataFrame;

template<typename T>
void GetDataTypes(std::string file_name, std::vector<std::vector<std::string> > &names_types,int &n_rows, int &n_columns, bool header=false);

template<typename T>
void ReadCSV(std::string file_name, MyDataFrame &df, std::vector<std::string> &column_names_string, bool header = false);

void ReadFile(std::string file_name, std::string &data);
void ReadCSVZIP(std::string file_name, std::string &data);



void SaveData(std::string file_path, std::string name, const std::map<std::string, std::map<boost::gregorian::date, std::vector<long double> > > &data);

std::string DateTimeToString(boost::posix_time::ptime datetime);


}

#endif /* SAVEDATA_H_ */
