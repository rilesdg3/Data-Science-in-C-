/*
 * Data.cpp
 *
 *  Created on: Mar 18, 2019
 *      Author: ryan
 */

#include "Data.h"
#include "Stats.h"


using namespace std;


/* @brief
 * A collection of Functions that converts data
 * into format/container type needed
 */

namespace MyData{

//if this is defined will use functions for multimaps instead of vectors were it matters
//#define MAP 1


void EmbedVect(std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd,
		std::vector<std::vector<double> > &vect, int vect_data_it, int m, int d, bool skip){


	std::multimap<double, std::vector<float>,classcomp>::iterator iter;

	cout<<"EmbedVect cBegin "<<cBegin->first<<" cEnd "<<cEnd->first<<endl;
	int end = std::abs(m*d)-d+1;

		//int vect_size = data.size() - end;

	int vect_data_it_start = vect_data_it;

		//int vect_data_it =0;
		//boost::timer::auto_cpu_timer t;

		if(skip == false){
			for( ; cEnd!=cBegin; cEnd--){
				iter = cEnd;

				//cout<<std::distance(cBegin, cEnd)<<endl;

				for(int i =1; i<m; i++){
					std::advance(iter, -1*std::abs(d));

					//cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
					//vect[m-i-1][vect_data_it] = iter->second[0];
					vect[vect_data_it][m-i-1] = iter->second[0];

					//vect[m-i-1].push_back(iter->second[0]);
					//cout<<"cEnd "<<cEnd->first<<" ";
					//cout<<"cEnd "<<cEnd->second<<" iter "<<iter->second<<endl;
					//cout<<"cEnd "<<cEnd->second<<" iter->second[0] "<<iter->second[0]<<endl;
					//cout<<"cEnd "<<cEnd->second<<" iter->second[1] "<<iter->second[1]<<endl;

					//cout<<"vect[m-i-1] "<<m-i-1<<" values "<<vect[m-i-1]<<endl;
				}
				if(std::distance(cBegin, cEnd)<end){
				//	data.erase(cBegin, cEnd);
					break;
				}
				vect_data_it++;
				//if(vect_data_it>=m+vect_data_it_start){
				//	break;
				//}
			}
		}
		else{
			for( ; cEnd!=cBegin; cEnd--){
				iter = cEnd;
				//cout<<std::distance(cBegin, cEnd)<<endl;

				for(int i =1; i<m; i++){
					std::advance(iter, -1*std::abs(d));
					//cout<<"cEnd b4 "<<cEnd->second<<" iter "<<iter->second<<endl;
					cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
				}
				cEnd->second.pop_back();

				if(std::distance(cBegin, cEnd)<end){
				//	data.erase(cBegin, cEnd);
					break;
				}
			}
		}
		//Print(data);
		cout<<"vect_data_it_start "<<vect_data_it_start<<" vect_data_it "<<vect_data_it<<endl;

}


void EmbedVect1(std::vector<double>::iterator cBegin, std::vector<double>::iterator cEnd,
		std::vector<std::vector<double> > &vect, int vect_data_it, int m, int d, bool skip){


	std::vector<double>::iterator iter;//std::multimap<double, std::vector<float>,classcomp>::iterator iter;

	int end = std::abs(m*d)-d+1;


	cout<<"EmbedVect cBegin "<<*cBegin<<" cEnd "<<*cEnd<<endl;
	//int end = std::abs(m*d)-d+1;

		//int vect_size = data.size() - end;

	int vect_data_it_start = vect_data_it;

		//int vect_data_it =0;
		//boost::timer::auto_cpu_timer t;

		if(skip == false){
			for( ; cEnd!=cBegin; cEnd--){
				iter = cEnd;

				//cout<<std::distance(cBegin, cEnd)<<endl;

				for(int i =1; i<m; i++){
					std::advance(iter, -1*std::abs(d));

					//cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
					//vect[m-i-1][vect_data_it] = iter->second[0];
					vect[vect_data_it][m-i-1] = *iter;//->second[0];

					//vect[m-i-1].push_back(iter->second[0]);
					//cout<<"cEnd "<<cEnd->first<<" ";
					//cout<<"cEnd "<<cEnd->second<<" iter "<<iter->second<<endl;
					//cout<<"cEnd "<<cEnd->second<<" iter->second[0] "<<iter->second[0]<<endl;
					//cout<<"cEnd "<<cEnd->second<<" iter->second[1] "<<iter->second[1]<<endl;

					//cout<<"vect[m-i-1] "<<m-i-1<<" values "<<vect[m-i-1]<<endl;
				}
				if(std::distance(cBegin, cEnd)<end){
				//	data.erase(cBegin, cEnd);
					break;
				}
				vect_data_it++;
				//if(vect_data_it>=m+vect_data_it_start){
				//	break;
				//}
			}
		}
		else{
/*			for( ; cEnd!=cBegin; cEnd--){
				iter = cEnd;
				//cout<<std::distance(cBegin, cEnd)<<endl;

				for(int i =1; i<m; i++){
					std::advance(iter, -1*std::abs(d));
					//cout<<"cEnd b4 "<<cEnd->second<<" iter "<<iter->second<<endl;
					cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
				}
				cEnd->second.pop_back();

				if(std::distance(cBegin, cEnd)<end){
				//	data.erase(cBegin, cEnd);
					break;
				}
			}*/
		}
		//Print(data);
		cout<<"vect_data_it_start "<<vect_data_it_start<<" vect_data_it "<<vect_data_it<<endl;

}

/*
 * @brief puts the most recent data(by time) at the end of the outer vector
 *
 *
 * note first mini of mutual inoframtion or lag # = 0 in acf should be the time lag tau
 *
 * d = tau = delay ???
 *
 * vect_starting_index is the index in the vector were you want the the first embeded value to go then
 * 			ex: index_to_start = 2,  vect[2] = first_value, vect[3] = second value ....
 *
 * mirrors the order of the data comin g in
 *
 */
void EmbedVectSameOrder(std::vector<double>::iterator cBegin, std::vector<double>::iterator cEnd,
		std::vector<std::vector<double> > &vect, int vect_data_it, int m, int d, int vect_starting_index, bool skip){


	std::vector<double>::iterator iter;//std::multimap<double, std::vector<float>,classcomp>::iterator iter;

	int end = std::abs(m*d)-d+1;


	cout<<"EmbedVect cBegin "<<*cBegin<<" cEnd "<<*cEnd<<endl;
	//int end = std::abs(m*d)-d+1;

		//int vect_size = data.size() - end;
	int count = 0;

	int vect_data_it_start = vect_data_it;

		//int vect_data_it =0;
		//boost::timer::auto_cpu_timer t;

		if(skip == false){
			for( ; cEnd!=cBegin; cEnd--){
				iter = cEnd;

				//cout<<std::distance(cBegin, cEnd)<<endl;
				//<= so we get exactly m days after the current day
				for(int i =vect_starting_index; i< m+vect_starting_index; i++){


					//cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
					//vect[m-i-1][vect_data_it] = iter->second[0];
					//vect[vect_data_it][m-i-1] = *iter;//->second[0];
					vect[vect_data_it][i] = *iter;//->second[0];
					std::advance(iter, -1*std::abs(d));

					//vect[m-i-1].push_back(iter->second[0]);
					//cout<<"cEnd "<<cEnd->first<<" ";
					//cout<<"cEnd "<<cEnd->second<<" iter "<<iter->second<<endl;
					//cout<<"cEnd "<<cEnd->second<<" iter->second[0] "<<iter->second[0]<<endl;
					//cout<<"cEnd "<<cEnd->second<<" iter->second[1] "<<iter->second[1]<<endl;

					//cout<<"vect[m-i-1] "<<m-i-1<<" values "<<vect[m-i-1]<<endl;
				}
				if(std::distance(cBegin, cEnd)<end){
				//	data.erase(cBegin, cEnd);
					break;
				}
				--vect_data_it;
				++count;
				//if(vect_data_it>=m+vect_data_it_start){
				//	break;
				//}
			}
		}
		else{
/*			for( ; cEnd!=cBegin; cEnd--){
				iter = cEnd;
				//cout<<std::distance(cBegin, cEnd)<<endl;

				for(int i =1; i<m; i++){
					std::advance(iter, -1*std::abs(d));
					//cout<<"cEnd b4 "<<cEnd->second<<" iter "<<iter->second<<endl;
					cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
				}
				cEnd->second.pop_back();

				if(std::distance(cBegin, cEnd)<end){
				//	data.erase(cBegin, cEnd);
					break;
				}
			}*/
		}
		//Print(data);
		cout<<"vect_data_it_start "<<vect_data_it_start<<" vect_data_it "<<vect_data_it<<endl;

}

/*
 * @brief mirrors the order of the data coming
 * 		the very first value is counted in m so if m == 25 then there will be 25 values
 * 		so date 0 will be the first then date 0+d will be the second
 *
 *
 * vect_starting_index is the index in the vector were you want the the first embeded value to go then
 * 			ex: index_to_start = 2,  vect[2] = first_value, vect[3] = second value ....
 *
 *
 *
 *
 */
template<typename T, typename T1>
std::multimap<T1,std::vector<T> > EmbedVectSameOrder(MyDataFrame &df, std::string variable_column_name, std::string time_date_column_name,
		int total_number_of_features, int m, int d,int vect_starting_index, bool skip){


	std::vector<T1> time_values = df.get_column<T1>(time_date_column_name.c_str());
	int vect_data_it = time_values.size()-1;
	std::vector<T> values = df.get_column<T>(variable_column_name.c_str());
	typename std::vector<T>::iterator cBegin = values.begin();
	typename std::vector<T>::iterator cEnd = values.end();
	std::multimap<T1, std::vector<T> > embeded_data;
	typename std::multimap<T1, std::vector<T> >::iterator embeded_data_iter;
	std::vector<double>::iterator iter;//std::multimap<double, std::vector<float>,classcomp>::iterator iter;

	//total_number_of_features = m+12/*months*/+6/*TA_Indicators*/-1;
	int vect_size = total_number_of_features;//end;//distance - end;//data.size() - end;
	std::vector<double > vect_data (vect_size,0);

	int end = std::abs(m*d)-d+1;
	int distance = std::distance(values.begin(),values.end());
	int outer_vect_size= distance - end;


	--cEnd;
	//cout<<"EmbedVect cBegin "<<*cBegin<<" cEnd "<<*cEnd<<endl;
	//int end = std::abs(m*d)-d+1;

		//int vect_size = data.size() - end;
	int count = 0;

	int vect_data_it_start = vect_data_it;

		//int vect_data_it =0;
		//boost::timer::auto_cpu_timer t;

		if(skip == false){
			for( ; cEnd!=cBegin; cEnd--){
				iter = cEnd;

				//cout<<std::distance(cBegin, cEnd)<<endl;
				std::vector<T> embeded_vect(total_number_of_features,0);

				embeded_data_iter = embeded_data.insert(embeded_data.begin(), std::pair<T1, std::vector<T> >(time_values[vect_data_it] ,embeded_vect) );


										//<= so we get exactly m days after the current day
				for(int i =vect_starting_index; i<m+vect_starting_index; i++){

					//std::cout<<embeded_data_iter->first<<" time_values[vect_data_it] "<<time_values[vect_data_it]<<std::endl;
					embeded_data_iter->second[i] = *iter;
					std::advance(iter, -1*std::abs(d));
				}
				if(std::distance(cBegin, cEnd)<end){
				//	data.erase(cBegin, cEnd);
					break;
				}
				--vect_data_it;
				++count;
			}
		}
		else{
/*			for( ; cEnd!=cBegin; cEnd--){
				iter = cEnd;
				//cout<<std::distance(cBegin, cEnd)<<endl;

				for(int i =1; i<m; i++){
					std::advance(iter, -1*std::abs(d));
					//cout<<"cEnd b4 "<<cEnd->second<<" iter "<<iter->second<<endl;
					cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
				}
				cEnd->second.pop_back();

				if(std::distance(cBegin, cEnd)<end){
				//	data.erase(cBegin, cEnd);
					break;
				}
			}*/
		}
		//Print(data);
		//cout<<"vect_data_it_start "<<vect_data_it_start<<" vect_data_it "<<vect_data_it<<endl;

		return embeded_data;

}
template std::multimap<boost::gregorian::date, std::vector<double> > EmbedVectSameOrder<double, boost::gregorian::date>(MyDataFrame &, std::string, std::string, int , int, int, int, bool );




/*
 * @brief: performs time delay embedding using multiple threads
 *
 * @param data:
 * @param int m: number of embedding dimesnions
 * @param int d: time delay
 * @param bool skip: whether to skip the first value, this is used when when using same data but with different time delays
 *
 */
std::vector<std::vector<double> > EmbedThreading(std::multimap<double, std::vector<float>, classcomp >  &data ,int m, int d, bool skip,
		std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd,bool pnt_data ){


	/*std::multimap<double, std::vector<float>,classcomp>::iterator test;
	if(cBegin != data.begin()){
		cBegin = data.begin();
		cEnd = data.end();
	}*/

	std::multimap<double, std::vector<float>,classcomp>::iterator iter;
	//cEnd--;

	int end;

	if(pnt_data == true)
		end = std::abs(m*d)-d+1;
	else
		end = std::abs(m*d)-d;

	int distance = std::distance(cBegin,cEnd);

	int vect_size = end;//distance - end;//data.size() - end;
	std::vector<double > vect_data (vect_size);
	std::vector<std::vector<double> > vect(distance - end);//m-1);


	for(uint i = 0; i< vect.size(); ++i)
		vect[i] = vect_data;

	int n_cores = std::thread::hardware_concurrency();


	//cout<<"modulus "<<std::modulus<double>((double)vect_size/(double)n_cores)<<endl;
	cout<<"modulus "<<vect_size/n_cores<<endl;
	cout<<"modulus "<<vect_size%n_cores<<endl;

	std::div_t map_it_per_core= std::div(distance,n_cores);//std::div(data.size(),n_cores);
	//map_it_per_core();
	//data.size()/n_cores;
	//int iter_per_core = vect_size/n_cores;
	int iter_per_core = (distance - end)/n_cores;

	std::vector<std::thread> threads;

	int what_to_name = 0;//vect_size;

	iter = cEnd;
	cBegin = cEnd;

	for(int i = 0; i< n_cores; ++i){

		std::advance(cBegin, -map_it_per_core.quot);


		cout<<"i "<<i<<" cBegin "<<cBegin->first<<" iter "<<iter->first;
		cout<<" what_to_name "<<what_to_name<<endl;
		//if(cBegin == data.end()){
		//	cBegin++;
		//	cout<<"i "<<i<<" cBegin "<<cBegin->first<<" iter "<<iter->first;
		//}
		if(iter == data.end())
			iter--;
		threads.push_back(std::thread(EmbedVect, cBegin, iter, std::ref(vect), what_to_name, m, d,skip));
		what_to_name = what_to_name + iter_per_core;
		std::advance(iter, -map_it_per_core.quot);
		cout<<"iter after advance "<<iter->first<<endl;
		//if(iter == data.end())
		//	break;

	}

	for(uint i = 0; i<threads.size(); ++i)
		{
		threads[i].join();
		cout<<"i "<<i<<endl;
		}




	return vect;
}

/*
 * @brief: performs time delay embedding returning a vector of vectors were the inside vector contains the time delayed embeding data
 *
 * @param data:
 * @param int m: number of embedding dimesnions
 * @param int d: time delay
 * @param bool skip: whether to skip the first value, this is used when when using same data but with different time delays
 *
 * @return std::vector<std::vector<double> > :
 *
 */
std::vector<std::vector<double> > Embed(std::multimap<double, std::vector<float>, classcomp >  &data ,int m, int d, bool skip){


	std::multimap<double, std::vector<float>,classcomp>::iterator cBegin = data.begin();
	std::multimap<double, std::vector<float>,classcomp>::iterator cEnd = data.end();
	std::multimap<double, std::vector<float>,classcomp>::iterator iter;
	cEnd--;

	int end = std::abs(m*d)-d+1;

	int vect_size = data.size() - end;
	std::vector<double > vect_data (vect_size);
	std::vector<std::vector<double> > vect(m-1);

	for(int i = 0; i< vect.size(); ++i)
		vect[i] = vect_data;

	int n_cores = std::thread::hardware_concurrency();

	//cout<<"modulus "<<std::modulus<double>((double)vect_size/(double)n_cores)<<endl;
	cout<<"modulus "<<vect_size/n_cores<<endl;
	cout<<"modulus "<<vect_size%n_cores<<endl;

	int iter_per_core = vect_size/n_cores;


	int vect_data_it =0;
	//boost::timer::auto_cpu_timer t;
	if(skip == false){
		for( ; cEnd!=cBegin; cEnd--){
			iter = cEnd;
			//cout<<std::distance(cBegin, cEnd)<<endl;

			for(int i =1; i<m; i++){
				std::advance(iter, -1*std::abs(d));

				//cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
				vect[m-i-1][vect_data_it] = iter->second[0];
				//vect[m-i-1].insert(vect[m-i-1].begin(),iter->second[0]);



				//vect[m-i-1].push_back(iter->second[0]);
				//cout<<"cEnd "<<cEnd->first<<" ";
				//cout<<"cEnd "<<cEnd->second<<" iter "<<iter->second<<endl;
				//cout<<"cEnd "<<cEnd->second<<" iter->second[0] "<<iter->second[0]<<endl;
				//cout<<"cEnd "<<cEnd->second<<" iter->second[1] "<<iter->second[1]<<endl;

				//cout<<"vect[m-i-1] "<<m-i-1<<" values "<<vect[m-i-1]<<endl;
			}
			if(std::distance(cBegin, cEnd)<end){
				data.erase(cBegin, cEnd);
				break;
			}
			vect_data_it++;
			//if(vect_data_it>750)
			//	break;
		}
	}
	else{
		for( ; cEnd!=cBegin; cEnd--){
			iter = cEnd;
			//cout<<std::distance(cBegin, cEnd)<<endl;

			for(int i =1; i<m; i++){
				std::advance(iter, -1*std::abs(d));
				//cout<<"cEnd b4 "<<cEnd->second<<" iter "<<iter->second<<endl;
				cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
			}
			cEnd->second.pop_back();

			if(std::distance(cBegin, cEnd)<end){
				data.erase(cBegin, cEnd);
				break;
			}
		}
	}
	//Print(data);

	for(int i = 0; i<vect.size(); ++i){
		vect[i].resize(vect_data_it);
		vect[i].shrink_to_fit();
	}


	return vect;
}


/*
 * @brief: Aligns Y with the corresponding embeded x values
 *
 * @param std::multimap<double, std::vector<float>, classcomp >  &data: map of time delayed x values
 *
 * @return std::vector<double> : a vector were the value is lined up with the corresponding time delayed x values
 *
 *
 */
std::vector<double> AlignedYvectWithEmbedX(std::multimap<double, std::vector<float>, classcomp >  &data,
		std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd){


	//std::multimap<double, std::vector<float>,classcomp>::iterator cBegin = data.begin();
	//std::multimap<double, std::vector<float>,classcomp>::iterator cEnd = data.end();
	std::multimap<double, std::vector<float>,classcomp>::iterator iter;

	int distance = std::distance(cBegin,cEnd);
	std::vector<double > vect_y;//(distance);//corresponds with embeded data were the y's should correspond with the last vector in vect and the last index
	//in data->second.size()-1

	//if(cEnd == data.end())
		cEnd--;
	int i = vect_y.size()-1;
	for(; cEnd!=cBegin; --cEnd,--i){
		//vect_y[i]=cEnd->first;//
		vect_y.push_back(cEnd->first);
		//--i;
	}

	return vect_y;

}


/*
 * NOTE: data must be a string and this will convert it to numeric as it groups by
 *
 * @brief: uses a map to put data into groups, were each group can have multiple sets of variables associated with it. were one variable
 * represents an index to a set of strings, in short this creates a one hot encoding for that variable
 *
 * @tparam set<T> &set_2_convert: A set of variables that need to be converted
 * @tparam vector<vector<T1> &data: The input data that contains the variables
 * @tparam T2 &grouped: The container that we are putting the grouped by and converted into
 * @param int group_by_column: The column number of variable that we are grouping by
 * @param int set_column: The column number in the data vector for the data found in groups that we are converting
 * @param set<int> var_columns: The columns to be used as variables
 *
 */
template< typename T, typename T1, typename T2, typename T3 >
void GroupBy(std::set<T> &set_2_convert,std::vector<std::vector<T1> > &data, T2 &grouped, int group_by_column, int set_column, T3 &var_columns){

	//auto set_it = set_2_convert.begin();

	string group_by_string;

	decltype(grouped.begin()->second) tmp_vect;

	int     status;
	char   *realname;
	//const std::type_info  &ti = typeid(tmp_vect.data());
	const std::type_info  &ti = typeid(tmp_vect);
	realname = abi::__cxa_demangle(ti.name(), 0, 0, &status);
	cout<<realname<<endl;
	string type = realname;

	int count=0;

	if(string(realname).find("double")!= std::string::npos){

		std::vector<double> tmp_vect(var_columns.size());

		//uint data_it = 0; data_it <data.size(); ++data_it
		for(auto data_it : data ){
			for(auto it = var_columns.begin(); it!= var_columns.end(); ++it){
				if(*it != group_by_column){
					if(*it == set_column)
						tmp_vect[count]=(double)std::distance(set_2_convert.begin(),set_2_convert.find(data_it[*it]));
					else
						tmp_vect[count] = std::stod(data_it[*it]);
				}
				count++;
			}
			grouped[data_it[group_by_column]].push_back(tmp_vect);
			count=0;
		}
	}




}
template void GroupBy<string, string, map<string, vector<vector<double> > > >(std::set<string> &, std::vector<std::vector<string > > &, map<string, vector<vector< double> > > &, int,int,std::set<int> &);
template void GroupBy<string, string, map<string, vector<vector<double> > > >(std::set<string> &, std::vector<std::vector<string > > &, map<string, vector<vector< double> > > &, int,int,std::unordered_set<int> &);
template void GroupBy<string, string, map<string, vector<vector<double> > > >(std::set<string> &, std::vector<std::vector<string > > &, map<string, vector<vector< double> > > &, int,int,vector<int> &);

/*
 * NOTE data variable must be all numeric
 *
 * @brief: uses a map to put data into groups, were each group can have multiple sets of variables associated with it. were one variable
 * represents an index to a set of strings, in short this creates a one hot encoding for that variable
 *
 * @tparam set<T> &set_2_convert: A set of variables that need to be converted
 * @tparam vector<vector<T1> &data: The input data that contains the variables
 * @tparam T2 &grouped: The container that we are putting the grouped by and converted into
 * @param int group_by_column: The column number of variable that we are grouping by
 * @param int set_column: The column number in the data vector for the data found in groups that we are converting
 * @param set<int> var_columns: The columns to be used as variables
 *
 */
template< typename T, typename T1, typename T2, typename T3 >
void GroupByAllNumeric(std::set<T> &set_2_convert,std::vector<std::vector<T1> > &data, T2 &grouped, int group_by_column, int set_column, T3 &var_columns){

	//auto set_it = set_2_convert.begin();

	string group_by_string;

	decltype(grouped.begin()->second) tmp_vect;

	int     status;
	char   *realname;
	//const std::type_info  &ti = typeid(tmp_vect.data());
	const std::type_info  &ti = typeid(tmp_vect);
	realname = abi::__cxa_demangle(ti.name(), 0, 0, &status);
	cout<<realname<<endl;
	string type = realname;

	int count=0;

	if(string(realname).find("double")!= std::string::npos){

		std::vector<T1> tmp_vect(var_columns.size());

		//uint data_it = 0; data_it <data.size(); ++data_it
		for(auto data_it : data ){
			for(auto it = var_columns.begin(); it!= var_columns.end(); ++it){
				if(*it != group_by_column){
					if(*it == set_column)
						tmp_vect[count]=(double)std::distance(set_2_convert.begin(),set_2_convert.find(data_it[*it]));
					else
						tmp_vect[count] =data_it[*it];
				}
				count++;
			}
			grouped[data_it[group_by_column]].push_back(tmp_vect);
			count=0;
		}
	}




}
template void GroupByAllNumeric<long double, long double, map<long double, vector<vector<long double> > > >(std::set<long double> &, std::vector<std::vector<long double > > &, map<long double, vector<vector< long double> > > &, int,int,vector<int> &);




/*
 * @brief: converts everything into a map<type, vector<double> >
 *
 * @tparam T:
 * @tprarm T1: map to put the data
 *
 */
template< typename T, typename T1 >
void GroupBy(T &data, T1 &grouped, int group_by_column, int value_column){


	string type;
	double value;

	for(uint data_it = 0; data_it <data.size(); ++data_it){
		type = data[data_it][group_by_column];
		value = std::stod(data[data_it][value_column]);
		grouped[data[data_it][group_by_column]].push_back(std::stod(data[data_it][value_column]));


	}



}
template void GroupBy<std::vector<std::vector<string > >, std::map<string, std::vector<double> > >( std::vector<std::vector<string > > &, std::map<string, std::vector<double> >  &, int,int);


/*
 * @brief: Converts column in a vector<vector<string> > to a single vector<float>
 *
 * @param vector<std::vector<string> >: &data_the data
 * @param int colum: column number of data to be converted to float
 *
 */
std::vector<float> VectorVectorStringToVectorFloat(std::vector<std::vector<string> > &data_in, int column){

	std::vector<float> tmp(data_in.size());

	for(int i = 0; i<data_in.size(); ++i)
		tmp[i]= std::stof(data_in[i][column]);

	return tmp;

}

/*
 * @brief: Takes features and labels data, splits it into Train, Validate, Test and allows for deciding what sections of the data will be used for
 * training, validation, and testing.
 *
 * @tparam T2, the features data
 * @tparam T3: the labels data
 * @param int TRTEVA: how to split up the data
 */
/*
template <typename T, typename T1>
template <typename T2, typename T3>
void SplitData<T, T1>::FnlDataToStruct(T2 &data, T3 &labels, int TRTEVA){





	int     status;
	char   *realname;
	//const std::type_info  &ti = typeid(tmp_vect.data());
	const std::type_info  &ti = typeid(data);
	realname = abi::__cxa_demangle(ti.name(), 0, 0, &status);
	cout<<realname<<endl;
	string type = realname;
	bool map = false;
	if(string(realname).find("map")!= std::string::npos)
		map = true;


#ifndef MAP
	this->FnlDataToStructMap(data,labels, TRTEVA);
#else
	this->FnlDataToStructVector(data,labels,TRTEVA);
#endif


}
#ifndef MAP
template void SplitData<std::multimap<boost::posix_time::ptime, std::vector<long double> >,std::vector<float > >::FnlDataToStruct<std::multimap<boost::posix_time::ptime, std::vector<long double> >,std::vector<float > >(std::multimap<boost::posix_time::ptime, std::vector<long double> > &,std::vector<float > &, int);
#else
template void SplitData<std::vector<std::vector<float > >,std::vector<float >  >::FnlDataToStruct<std::vector<std::vector<float > >,std::vector<float > >(std::vector<std::vector<float > > &, std::vector<float > &, int);
#endif
*/

#if DATA_MAP >= 1
/*
 * @brief: Takes features and labels data, splits it into Train, Validate, Test and allows for deciding what sections of the data will be used for
 * training, validation, and testing.
 *
 * @tparam T2, the features data
 * @tparam T3: the labels data
 * @param int TRTEVA: how to split up the data
 */
template <typename T, typename T1, typename Tg, typename Tm>
template<typename T2, typename T3>
void SplitData<T, T1, Tg, Tm>::FnlDataToStruct(T2 &data, T3 &labels, int TRTEVA, split_or_not split){



	int     status;
	char   *realname;
	//const std::type_info  &ti = typeid(tmp_vect.data());
	const std::type_info  &ti = typeid(labels);
	realname = abi::__cxa_demangle(ti.name(), 0, 0, &status);
	cout<<realname<<endl;
	string type = realname;

	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cBegin = data.begin();
	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cEnd = data.end();


	/*int nRows = data.size();

	int count = 0;
	int label = 0;

	//need to decide which of the two below to use
	//typename std::remove_reference<decltype(this->train_features_[0])>::type featureTemp;
	//std::vector<float> featureTemp;
	std::vector<int > labelTemp;

	int mylen=nRows-1;

	int trainamount = std::floor(mylen/3);
	int testamount = std::floor(mylen/3);
	int validamount= std::floor(mylen/3);

	std::pair<int,int> trainrange;
	std::pair<int,int> testrange;
	std::pair<int,int> validrange;

	int trainstart=0;
	int trainend=(trainstart+trainamount);
	int teststart=(trainend);
	int testend=(teststart+testamount);
	int validstart=(testend);
	int validend=(validstart+validamount);

	if(TRTEVA == 0){
		trainrange =std::make_pair(trainstart,trainend);
		testrange =std::make_pair(teststart,testend);
		validrange =std::make_pair(validstart,validend);
	}
	else if(TRTEVA == 1){
		trainrange=std::make_pair(teststart,testend);
		testrange=std::make_pair(trainstart,trainend);
		validrange=std::make_pair(validstart,validend);
	}
	else
	{
		trainrange=std::make_pair(validstart,validend);
		testrange=std::make_pair(trainstart,trainend);
		validrange=std::make_pair(teststart,testend);
	}


	for(; cBegin!=cEnd; cBegin++){
		auto vIt = cBegin->second.begin();
		auto vEnd = cBegin->second.end();
		vEnd--;
		for(; vIt!=vEnd; ++vIt)
		{
			featureTemp.push_back((float)*vIt);
		}
		label = (int)(std::trunc(*vIt));
		//if(count % 10 ==0 )
			//	label=2;
		//if(count %15 == 0)
		//	label =3;
		labelTemp.push_back(label);

		if(count>=trainrange.first && count<trainrange.second){
				TrainData.Labels.push_back(label);
				TrainData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				TrainData.Features.push_back(featureTemp);
			}
			else if(count>=testrange.first && count<testrange.second){
				TestData.Labels.push_back(label);
				TestData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				TestData.Features.push_back(featureTemp);
			}
			else {
				ValidateData.Labels.push_back(label);
				ValidateData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				ValidateData.Features.push_back(featureTemp);
			}

		AllData.DateToIter.push_back(cBegin->first);

		AllData.Labels.push_back(label);
		AllData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
		AllData.Features.push_back(featureTemp);

		featureTemp.clear();
		labelTemp.clear();
		count++;
	}*/




}
template void SplitData<std::vector<std::vector<float > >,std::vector<float > >::FnlDataToStruct<std::multimap<boost::posix_time::ptime, std::vector<long double> >,std::vector<float > >(std::multimap<boost::posix_time::ptime, std::vector<long double> > &, std::vector<float > &, int,split_or_not);
//template void SplitData<std::multimap<boost::posix_time::ptime, std::vector<long double> >,std::vector<float >  >::FnlDataToStruct<std::multimap<boost::posix_time::ptime, std::vector<long double> >,std::vector<float > >(std::multimap<boost::posix_time::ptime, std::vector<long double> > &, std::vector<float > &, int,split_or_not);



/*
 * @brief: creates iterators for the data by dividing the data into train, test and validate. All the data will stay in its container
 * 			and will be iterated over using the iterators that this function builds
 * 			Every image_id and class label should be combined to form an a unique name of the from "id_label"
 *
 * @tparam data: image data
 * @tparam labels; class labels
 * @tparam id_label_seperated: used to sep
 * @tparam masks:  the pixels that correspond to the label
 */
template <typename T, typename T1, typename Tg, typename Tm>
template<typename T2, typename T3, typename T4, typename T5>
void SplitData<T, T1,Tg, Tm>::BuildIters(T2 &data, T3 &labels, T4 &id_label_seperated, T5 &masks, int TRTEVA, split_or_not split){


	cout<<"BuildIters"<<endl;

	typedef const typename std::remove_reference<decltype(this->train_labels_)>::type tmp_vect_decoded_type;

	int     status;
			char   *realname;
			//const std::type_info  &ti = typeid(tmp_vect.data());
			const std::type_info  &ti = typeid(tmp_vect_decoded_type);
			realname = abi::__cxa_demangle(ti.name(), 0, 0, &status);
			cout<<realname<<endl;

	std::vector<int> labels_vect_hot_tmp(labels.size());
	std::vector<int> labels_vect_tmp(1);

	int nRows = this->to_get.size();//data.size();

	int count = 0;


	int mylen=nRows;//-1;

	int trainamount = std::floor(mylen/3);
	int testamount = std::floor(mylen/3);
	int validamount= std::floor(mylen/3);

	this->train_features_.resize(trainamount);
	this->test_features_.resize(testamount);
	this->validate_features_.resize(validamount);

	int train_features_count = 0;
	int	test_features_count = 0;
	int	validate_features_count = 0;

	std::pair<int,int> trainrange;
	std::pair<int,int> testrange;
	std::pair<int,int> validrange;

	int trainstart=0;
	int trainend=(trainstart+trainamount);
	int teststart=(trainend);
	int testend=(teststart+testamount);
	int validstart=(testend);
	int validend=(validstart+validamount);

	if(TRTEVA == 0){
		trainrange =std::make_pair(trainstart,trainend);
		testrange =std::make_pair(teststart,testend);
		validrange =std::make_pair(validstart,validend);
	}
	else if(TRTEVA == 1){
		trainrange=std::make_pair(teststart,testend);
		testrange=std::make_pair(trainstart,trainend);
		validrange=std::make_pair(validstart,validend);
	}
	else
	{
		trainrange=std::make_pair(validstart,validend);
		testrange=std::make_pair(trainstart,trainend);
		validrange=std::make_pair(teststart,testend);
	}

	string label = " ";
	string image_id = " ";



	//Todo; this for loop should go inside of the switch statements so I am not checking with every iteration
	//for(; cBegin!=cEnd; ++cBegin){
	for(uint i = 0; i< this->to_get.size(); ++i){

		image_id = id_label_seperated.find(this->to_get[i])->first;

		label = id_label_seperated.find(this->to_get[i])->second[1];

		int tmp_int = std::distance(labels.begin(), labels.find(label));

		labels_vect_tmp[0] = tmp_int;
		labels_vect_hot_tmp[tmp_int] = 1;

		switch(split){
		case this->no_split:{
			//this->all_data_labels_.push_back(labels_vect_tmp);//AllData.Labels.push_back(label);
			//this->all_data_labels_hot_.push_back(labels_vect_tmp);//AllData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
			//this->all_data_features_.push_back(featureTemp);//AllData.Features.push_back(featureTemp);
		}
		break;
		case this->split_data_only:
		{
			if(count>=trainrange.first && count<trainrange.second){

				//split_data.train_features_.push_back()
				this->train_labels_.insert(std::make_pair(image_id, labels_vect_tmp));///TrainData.Labels.push_back(label);
				this->train_labels_one_hot_.insert(std::make_pair(image_id, labels_vect_hot_tmp));//TrainData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);

				this->train_features_[train_features_count]=image_id;//this->train_features_.push_back(image_id);
				//train_features_itr = this->train_features_.find(image_id);

				++train_features_count;


			}
			else if(count>=testrange.first && count<testrange.second){
				this->test_labels_.insert(std::make_pair(image_id, labels_vect_tmp));//TestData.Labels.push_back(label);
				this->test_labels_one_hot_.insert(std::make_pair(image_id, labels_vect_hot_tmp));//TestData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->test_features_[test_features_count]=image_id;//TestData.Features.push_back(featureTemp);
				++test_features_count;
			}
			else if(count>=validrange.first && count<validrange.second)
			{
				this->validate_labels_.insert(std::make_pair(image_id, labels_vect_tmp));//ValidateData.Labels.push_back(label);
				this->validate_labels_one_hot_.insert(std::make_pair(image_id, labels_vect_hot_tmp));//ValidateData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->validate_features_[validate_features_count]=image_id;//ValidateData.Features.push_back(featureTemp);
				++validate_features_count;
			}
		}
		break;
		case this->split_and_all_data:
		{
			if(count>=trainrange.first && count<trainrange.second){

				//split_data.train_features_.push_back()
				this->train_labels_.insert(std::make_pair(image_id, labels_vect_tmp));///TrainData.Labels.push_back(label);
				this->train_labels_one_hot_.insert(std::make_pair(image_id, labels_vect_hot_tmp));//TrainData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->train_features_[train_features_count] =image_id;//TrainData.Features.push_back(featureTemp);
				++train_features_count;
			}
			else if(count>=testrange.first && count<testrange.second){
				this->test_labels_.insert(std::make_pair(image_id, labels_vect_tmp));//TestData.Labels.push_back(label);
				this->test_labels_one_hot_.insert(std::make_pair(image_id, labels_vect_hot_tmp));//TestData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->test_features_[test_features_count] = image_id;//TestData.Features.push_back(featureTemp);
				++test_features_count;
			}
			else if(count>=validrange.first && count<validrange.second)
			{
				this->validate_labels_.insert(std::make_pair(image_id, labels_vect_tmp));//ValidateData.Labels.push_back(label);
				this->validate_labels_one_hot_.insert(std::make_pair(image_id, labels_vect_hot_tmp));//ValidateData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->validate_features_[validate_features_count] = image_id;//ValidateData.Features.push_back(featureTemp);
				++validate_features_count;
			}
		}


		}


		labels_vect_hot_tmp[tmp_int] = 0.0;
		count++;
	}




}
template void SplitData<std::vector<string>,std::unordered_map<string,std::vector<int> > >::BuildIters<std::unordered_map<string, vector<float> >, std::set<string>, std::unordered_map<string, vector<string> >, std::unordered_map<string, vector<int> >  >(std::unordered_map<string, vector<float> > &, std::set<string> &, std::unordered_map<string, vector<string> > &, std::unordered_map<string, vector<int> > &,  int,split_or_not);
template void SplitData<std::vector<string>,std::unordered_map<string,std::vector<int> > >::BuildIters<std::unordered_map<string, vector<float> >, std::set<string>, std::unordered_map<string, vector<string> >, std::unordered_map<string, vector<float> >  >(std::unordered_map<string, vector<float> > &, std::set<string> &, std::unordered_map<string, vector<string> > &, std::unordered_map<string, vector<float> > &,  int,split_or_not);

//template void SplitData<std::vector<std::vector<float > >,std::vector< std::vector<float > >,std::vector<vector<int> >  >::FnlDataToStruct<std::multimap<string, vector<float> >,std::set<string>,std::multimap<string, std::multimap<string, vector<int> > > >(std::multimap<string, vector<float> > &, std::set<string> &, std::multimap<string, std::multimap<string, vector<int> > > &, int,split_or_not);


#else
/*
 * @brief: Takes features and labels data, splits it into Train, Validate, Test and allows for deciding what sections of the data will be used for
 * training, validation, and testing.
 *
 * @tparam T2, the features data
 * @tparam T3: the labels data
 * @param int TRTEVA: how to split up the data
 */
template <typename T, typename T1>
template<typename T2, typename T3>
void SplitData<T, T1>::FnlDataToStruct(T2 &data, T3 &labels, int TRTEVA, split_or_not split){


	std::vector<float> po;


	typename std::remove_reference<decltype(this->train_features_[0])>::type featureTemp;

	int cBegin = 0;//data.size();
	int cEnd = data.size();


	int nRows = data.size();

	int count = 0;
	float label = 0.0000000;

	int     status;
	char   *realname;
	//const std::type_info  &ti = typeid(tmp_vect.data());
	const std::type_info  &ti = typeid(label);
	realname = abi::__cxa_demangle(ti.name(), 0, 0, &status);
	cout<<realname<<endl;
	string type = realname;

	std::vector<float > labelTemp;

	int mylen=nRows-1;

	int trainamount = std::floor(mylen/3);
	int testamount = std::floor(mylen/3);
	int validamount= std::floor(mylen/3);

	std::pair<int,int> trainrange;
	std::pair<int,int> testrange;
	std::pair<int,int> validrange;

	int trainstart=0;
	int trainend=(trainstart+trainamount);
	int teststart=(trainend);
	int testend=(teststart+testamount);
	int validstart=(testend);
	int validend=(validstart+validamount);

	if(TRTEVA == 0){
		//0 == TRTEVA
		trainrange =std::make_pair(trainstart,trainend);
		testrange =std::make_pair(teststart,testend);
		validrange =std::make_pair(validstart,validend);
	}
	else if(TRTEVA == 1){
		//1 == TETRVA
		trainrange=std::make_pair(teststart,testend);
		testrange=std::make_pair(trainstart,trainend);
		validrange=std::make_pair(validstart,validend);
	}
	else
	{
		//2 == VATRTE
		trainrange=std::make_pair(validstart,validend);
		testrange=std::make_pair(trainstart,trainend);
		validrange=std::make_pair(teststart,testend);
	}


	for(; cBegin!=cEnd; cBegin++){
		auto vIt = 0;
		auto vEnd = data[cBegin].size();
		//vEnd--;
		for(; vIt!=vEnd; ++vIt)
		{
			featureTemp.push_back(data[cBegin][vIt]);//featureTemp.push_back((float)*vIt);
		}



		if(string(realname).find("int")!= std::string::npos)
			label = (float)(std::trunc(labels[cBegin]));
		else
			label = labels[cBegin];
		//if(count % 10 ==0 )
		//	label=2;
		//if(count %15 == 0)
		//	label =3;
		labelTemp.push_back(label);

		switch(split){
		case this->no_split:{
			this->all_data_labels_.push_back(label);//AllData.Labels.push_back(label);
			this->all_data_labels_hot_.push_back(labelTemp);//AllData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
			this->all_data_features_.push_back(featureTemp);//AllData.Features.push_back(featureTemp);
		}
		break;
		case this->split_data_only:
		{
			if(count>=trainrange.first && count<trainrange.second){

				//split_data.train_features_.push_back()
				this->train_labels_.push_back(label);///TrainData.Labels.push_back(label);
				//TrainData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->train_features_.push_back(featureTemp);//TrainData.Features.push_back(featureTemp);
			}
			else if(count>=testrange.first && count<testrange.second){
				this->test_labels_.push_back(label);//TestData.Labels.push_back(label);
				//TestData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->test_features_.push_back(featureTemp);//TestData.Features.push_back(featureTemp);
			}
			else {
				this->validate_labels_.push_back(label);//ValidateData.Labels.push_back(label);
				//ValidateData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->validate_features_.push_back(featureTemp);//ValidateData.Features.push_back(featureTemp);
			}
		}
		break;
		case this->split_and_all_data:
		{
			if(count>=trainrange.first && count<trainrange.second){

				//split_data.train_features_.push_back()
				this->train_labels_.push_back(label);///TrainData.Labels.push_back(label);
				//TrainData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->train_features_.push_back(featureTemp);//TrainData.Features.push_back(featureTemp);
			}
			else if(count>=testrange.first && count<testrange.second){
				this->test_labels_.push_back(label);//TestData.Labels.push_back(label);
				//TestData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->test_features_.push_back(featureTemp);//TestData.Features.push_back(featureTemp);
			}
			else {
				this->validate_labels_.push_back(label);//ValidateData.Labels.push_back(label);
				//ValidateData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
				this->validate_features_.push_back(featureTemp);//ValidateData.Features.push_back(featureTemp);
			}
			//AllData.Labels.push_back(label);
			//AllData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
			//AllData.Features.push_back(featureTemp);
		}


		}


		//AllData.Labels.push_back(label);
		//AllData.LabelsHot.push_back(labelTemp);//labelTemp.push_back(label);
		//AllData.Features.push_back(featureTemp);

		featureTemp.clear();
		labelTemp.clear();
		count++;
	}




}
template void SplitData<std::vector<std::vector<float > >,std::vector<float >  >::FnlDataToStruct<std::vector<std::vector<float > >,std::vector<float > >(std::vector<std::vector<float > > &, std::vector<float > &, int, split_or_not);

#endif


inline int ZeroOne(long double x){
	if(x<0)
		return 0;
	else if(x>=0)
		return 1;
	else{
		std::cout<<"Problem ZeroOne "<<std::endl;
		return 1000;
	}
}

inline int ZeroOneNegOne(long double x){
	if(x<0)
		return 0;
	else if(x>0)
		return 1;
	else if(x==0)
		return -1;
	else{
		std::cout<<"Problem: ZeroOneNegOne "<<std::endl;
		return 1000;
	}
}


#if GRETL == 1
/*
 *
 * @brief: converts a vector<T> into at GRETL DATASET struct
 *
 * @tparam vector<T> data: the to be put into DATASET->
 * @param DATASET *gretl_data_set: This is the data strut required for GRETL
 * @param int pd: frequency: ex: quarterly =4, monthly = 12, weekly = 52, daily = 5,6, or 7
 * @param int structure: panel = ??, time series = 1, cross sectional = ??, and ????
 * @param string variable_name: the name of the variable
 * @param bool include_constant_vector: if true puts a constant vector of 1's at dset->Z[0][i] = 1.00;
 * 										this is because dset->Z[0][] is reserved for a constant vector in gretl
 *
 */
template<typename T>
void ToGretl(std::vector<T> &data, DATASET *gretl_data_set, int pd, int structure, std::string variable_name, bool include_constant_vector){


	/*

	typedef struct {
	    int v;               number of variables
	    int n;               number of observations
	    int pd;              periodicity or frequency of data
	    int structure;       time series, cross section or whatever
	    double sd0;          float representation of stobs
	    int t1, t2;          start and end of current sample
	    char stobs[OBSLEN];   string representation of starting obs (date)
	    char endobs[OBSLEN];  string representation of ending obs
	    double **Z;          data array
	    char **varname;      array of names of variables
	    VARINFO **varinfo;   array of specific info on vars
	    char markers;        NO_MARKERS (0), REGULAR MARKERS or DAILY_DATE_STRINGS
	    char modflag;        binary flag for dataset modified or not
	    char **S;            to hold observation markers
	    char *descrip;       to hold info on data sources etc.
	    char *submask;       subsampling mask
	    char *restriction;   record of sub-sampling restriction
	    char *padmask;       record of padding to re-balance panel data
	    unsigned int rseed;  resampling seed
	    int auxiliary;       = 0 for regular dataset, 1 for aux dataset
	    char *pangrps;       panel-only: name of series holding group names
	    int panel_pd;        panel-only: panel time-series frequency
	    double panel_sd0;    panel-only: time-series start
	} DATASET;
	 */


	int nobs = data.size();
	gretl_data_set->pd = pd;
	gretl_data_set->structure = structure;//1=time series
	//gretl_data_set = create_new_dataset (1,nobs, 0);
	gretl_data_set->v = 1;//0;//# of vars +1 becuase the dataset requires a constant
	gretl_data_set->n = nobs;
	gretl_data_set->t1 = 0;//this should be the same as i in for loop below;
	gretl_data_set->t2 = nobs-1;//data.size();//this should be the same as i in for loop below;
	//char f_[] = variable_name.c_str();






	if(include_constant_vector == false){
		gretl_data_set->varname = (char **) malloc(sizeof(char *)*gretl_data_set->v);//malloc(f_* sizeof(f_));
		*gretl_data_set->varname = (char *) malloc(sizeof(char));//malloc(f_* size

		::dataset_allocate_varnames(gretl_data_set);
		int err = allocate_Z(gretl_data_set, OPT_NONE);
		for(int i = 0;  i<gretl_data_set->v; ++i){
			gretl_data_set->varname[i] = (char *) malloc(sizeof(char));
			strcpy(gretl_data_set->varname[i],  const_cast<char*>(variable_name.c_str()));
			::series_set_label(gretl_data_set, i, variable_name.c_str() );
			//::printf("%s\n",gretl_data_set->varname[i]);
		}


		if(gretl_data_set != NULL){

			int size_ = sizeof(gretl_data_set->Z)/sizeof(double);
			for(int i = 0; i<nobs;++i){
				gretl_data_set->Z[0][i] = data[i];
				//std::cout<<gretl_data_set->Z[0][i]<<" i "<<i<<" nobs "<<nobs<<std::endl;
			}

		}
	}

	else{
		gretl_data_set->v = 2;
		gretl_data_set->varname = (char **) malloc(sizeof(char *)*gretl_data_set->v);//malloc(f_* sizeof(f_));
		*gretl_data_set->varname = (char *) malloc(sizeof(char));//malloc(f_* size

		::dataset_allocate_varnames(gretl_data_set);
		int err = allocate_Z(gretl_data_set, OPT_NONE);


		gretl_data_set->varname[0] = (char *) malloc(sizeof(char));
		strcpy(gretl_data_set->varname[0],  "constant");
		::series_set_label(gretl_data_set, 0, "constant");
		for(int i = 1;  i<gretl_data_set->v; ++i){
			gretl_data_set->varname[i] = (char *) malloc(sizeof(char));
			strcpy(gretl_data_set->varname[i],  const_cast<char*>(variable_name.c_str()));
			::series_set_label(gretl_data_set, i, variable_name.c_str() );
			//::printf("%s\n",gretl_data_set->varname[i]);
		}


		if(gretl_data_set != NULL){

			int size_ = sizeof(gretl_data_set->Z)/sizeof(double);
			for(int i = 0; i<nobs;++i){
				gretl_data_set->Z[0][i] = 1.0000;
				gretl_data_set->Z[1][i] = data[i];
				//std::cout<<gretl_data_set->Z[0][i]<<" i "<<i<<" nobs "<<nobs<<std::endl;
			}




		}
	}


}
template void ToGretl<double>(std::vector<double> &, DATASET *, int , int , std::string,bool );

/*
 *
 * @brief: converts a vector<T> into at GRETL DATASET struct
 *
 * @tparam std::vector<vector<T> > data: the to be put into DATASET->
 * @param DATASET *gretl_data_set: This is the data strut required for GRETL
 * @param int pd: frequency: ex: quarterly =4, montly = 12, weekly = 52, daily = 5,6, or 7
 * @param int structure: panel = ??, time series = 1, cross sectional = ??, and ????
 * @param vector<string> variable_name: the names of the variables
 * @param bool include_constant_vector: if true puts a constant vector of 1's at dset->Z[0][i] = 1.00;
 * 										this is because dset->Z[0][] is reserved for a constant vector in gretl
 *
 */
template<typename T>
void ToGretl(std::vector<std::vector<T> > &data, DATASET *gretl_data_set, int pd, int structure, std::vector<std::string> variable_name, bool include_constant_vector){


	/*

	typedef struct {
	    int v;               number of variables
	    int n;               number of observations
	    int pd;              periodicity or frequency of data
	    int structure;       time series, cross section or whatever
	    double sd0;          float representation of stobs
	    int t1, t2;          start and end of current sample
	    char stobs[OBSLEN];   string representation of starting obs (date)
	    char endobs[OBSLEN];  string representation of ending obs
	    double **Z;          data array
	    char **varname;      array of names of variables
	    VARINFO **varinfo;   array of specific info on vars
	    char markers;        NO_MARKERS (0), REGULAR MARKERS or DAILY_DATE_STRINGS
	    char modflag;        binary flag for dataset modified or not
	    char **S;            to hold observation markers
	    char *descrip;       to hold info on data sources etc.
	    char *submask;       subsampling mask
	    char *restriction;   record of sub-sampling restriction
	    char *padmask;       record of padding to re-balance panel data
	    unsigned int rseed;  resampling seed
	    int auxiliary;       = 0 for regular dataset, 1 for aux dataset
	    char *pangrps;       panel-only: name of series holding group names
	    int panel_pd;        panel-only: panel time-series frequency
	    double panel_sd0;    panel-only: time-series start
	} DATASET;
	 */


	int nobs = data[0].size();
	for(int i = 0; i<data.size(); ++i){
		if(data[i].size() > nobs)
			nobs=data[i].size();
	}

	gretl_data_set->pd = pd;
	gretl_data_set->structure = structure;//1=time series
	//gretl_data_set = create_new_dataset (1,nobs, 0);
	gretl_data_set->v = variable_name.size();//data.size();//0;//# of vars +1 becuase the dataset requires a constant
	gretl_data_set->n = nobs;
	gretl_data_set->t1 = 0;//this should be the same as i in for loop below;
	gretl_data_set->t2 = nobs-1;//data.size();//this should be the same as i in for loop below;
	//char f_[] = variable_name.c_str();



	if(include_constant_vector == false){
		gretl_data_set->varname = (char **) malloc(sizeof(char *)*gretl_data_set->v);//malloc(f_* sizeof(f_));
		*gretl_data_set->varname = (char *) malloc(sizeof(char));//malloc(f_* size


		::dataset_allocate_varnames(gretl_data_set);
		int err = allocate_Z(gretl_data_set, OPT_NONE);
		for(int i = 0;  i<gretl_data_set->v; ++i){
			gretl_data_set->varname[i] = (char *) malloc(sizeof(char));
			strcpy(gretl_data_set->varname[i],  const_cast<char*>(variable_name[i].c_str()));
			::series_set_label(gretl_data_set, i, variable_name[i].c_str() );
			//::printf("%s\n",gretl_data_set->varname[i]);
		}

		if(gretl_data_set != NULL){
			for(int j = 0; j<data.size();++j){
				for(int i = 0; i<nobs;++i){

					if(i > data[j].size()-1){
						gretl_data_set->Z[j][i] = NADBL;
					}
					else
						gretl_data_set->Z[j][i] = data[j][i];

					//std::cout<<gretl_data_set->Z[j][i]<<" j "<<j<<" number of vars "<<data.size()<<" i "<<i<<" nobs "<<nobs<<std::endl;
				}
			}

		}
	}
	else{

		//gretl_data_set->v=data.size();
		gretl_data_set->varname = (char **) malloc(sizeof(char *)*gretl_data_set->v);//malloc(f_* sizeof(f_));
		*gretl_data_set->varname = (char *) malloc(sizeof(char));//malloc(f_* size


		::dataset_allocate_varnames(gretl_data_set);
		int err = allocate_Z(gretl_data_set, OPT_NONE);
		/*gretl_data_set->varname[0] = (char *) malloc(sizeof(char));
		strcpy(gretl_data_set->varname[0],  "constant");
		::series_set_label(gretl_data_set, 0, "constant")*/;

		for(int i = 0;  i<gretl_data_set->v; ++i){
			gretl_data_set->varname[i] = (char *) malloc(sizeof(char));
			strcpy(gretl_data_set->varname[i],  const_cast<char*>(variable_name[i].c_str()));
			::series_set_label(gretl_data_set, i, variable_name[i].c_str() );
			//::printf("%s\n",gretl_data_set->varname[i]);
		}

		if(gretl_data_set != NULL){
			for(int i = 0; i<nobs;++i)
				gretl_data_set->Z[0][i] = 1.0000;

			for(int j = 0; j<data.size();++j){
				for(int i = 0; i<nobs;++i){

					if(i > data[j].size()-1){
						gretl_data_set->Z[j+1][i] = NADBL;
					}
					else
						gretl_data_set->Z[j+1][i] = data[j][i];

					//std::cout<<gretl_data_set->Z[j][i]<<" j "<<j<<" number of vars "<<data.size()<<" i "<<i<<" nobs "<<nobs<<std::endl;
				}
			}

		}
	}


}
template void ToGretl<double>(std::vector<std::vector<double> > &, DATASET *, int , int , std::vector<std::string> ,bool );

/*
 *
 * @brief: converts a vector<T> into at GRETL DATASET struct
 *
 * @tparam std::vector<vector<T> > data: the to be put into DATASET->
 * @param DATASET *gretl_data_set: This is the data strut required for GRETL
 * @param int pd: frequency: ex: quarterly =4, montly = 12, weekly = 52, daily = 5,6, or 7
 * @param int structure: panel = ??, time series = 1, cross sectional = ??, and ????
 * @param vector<string> variable_name: the names of the variables
 * @param bool include_constant_vector: if true puts a constant vector of 1's at dset->Z[0][i] = 1.00;
 * 										this is because dset->Z[0][] is reserved for a constant vector in gretl
 *
 */
template<typename T>
void ToGretl(MyDataFrame &data, DATASET *gretl_data_set, int pd, int structure, std::vector<std::string> variable_name, bool include_constant_vector){


	/*

	typedef struct {
	    int v;               number of variables
	    int n;               number of observations
	    int pd;              periodicity or frequency of data
	    int structure;       time series, cross section or whatever
	    double sd0;          float representation of stobs
	    int t1, t2;          start and end of current sample
	    char stobs[OBSLEN];   string representation of starting obs (date)
	    char endobs[OBSLEN];  string representation of ending obs
	    double **Z;          data array
	    char **varname;      array of names of variables
	    VARINFO **varinfo;   array of specific info on vars
	    char markers;        NO_MARKERS (0), REGULAR MARKERS or DAILY_DATE_STRINGS
	    char modflag;        binary flag for dataset modified or not
	    char **S;            to hold observation markers
	    char *descrip;       to hold info on data sources etc.
	    char *submask;       subsampling mask
	    char *restriction;   record of sub-sampling restriction
	    char *padmask;       record of padding to re-balance panel data
	    unsigned int rseed;  resampling seed
	    int auxiliary;       = 0 for regular dataset, 1 for aux dataset
	    char *pangrps;       panel-only: name of series holding group names
	    int panel_pd;        panel-only: panel time-series frequency
	    double panel_sd0;    panel-only: time-series start
	} DATASET;
	 */



	//find vector with largest size

	int nobs = 0;//data[0].size();
	/*	for(int i = 0; i<data.size(); ++i){
		if(data[i].size() > nobs)
			nobs=data[i].size();
	}
	 */
	//data.get_columns_info()
	for(auto i : variable_name){
		if(data.get_column<double>(i.c_str()).size()>nobs)
			nobs =data.get_column<double>(i.c_str()).size();
	}

	gretl_data_set->pd = pd;
	gretl_data_set->structure = structure;//1=time series
	//gretl_data_set = create_new_dataset (1,nobs, 0);
	gretl_data_set->v = variable_name.size();//data.size();//0;//# of vars +1 becuase the dataset requires a constant
	gretl_data_set->n = nobs;
	gretl_data_set->t1 = 0;//this should be the same as i in for loop below;
	gretl_data_set->t2 = nobs-1;//data.size();//this should be the same as i in for loop below;
	//char f_[] = variable_name.c_str();



	if(include_constant_vector == false){
		gretl_data_set->varname = (char **) malloc(sizeof(char *)*gretl_data_set->v);//malloc(f_* sizeof(f_));
		*gretl_data_set->varname = (char *) malloc(sizeof(char));//malloc(f_* size


		::dataset_allocate_varnames(gretl_data_set);
		int err = allocate_Z(gretl_data_set, OPT_NONE);
		for(int i = 0;  i<gretl_data_set->v; ++i){
			gretl_data_set->varname[i] = (char *) malloc(sizeof(char));
			strcpy(gretl_data_set->varname[i],  const_cast<char*>(variable_name[i].c_str()));
			::series_set_label(gretl_data_set, i, variable_name[i].c_str() );
			//::printf("%s\n",gretl_data_set->varname[i]);
		}

		if(gretl_data_set != NULL){
			int count = 0;
			//for(int j = 0; j<data.size();++j){
			for(auto j : variable_name){
				//data.get_column<double>(j.c_str)[i]
				for(int i = 0; i<nobs;++i){

					if(i > data.get_column<double>(j.c_str()).size()-1 || std::isnan(data.get_column<double>(j.c_str())[i])){
						gretl_data_set->Z[count][i] = NADBL;
					}
					else
						gretl_data_set->Z[count][i] = data.get_column<double>(j.c_str())[i];

					//std::cout<<gretl_data_set->Z[count][i]<<" count "<<count<<" i "<<i<<j<<std::endl;
				}
				++count;
			}
		}
	}
	else{

		gretl_data_set->v=variable_name.size();
		gretl_data_set->varname = (char **) malloc(sizeof(char *)*gretl_data_set->v);//malloc(f_* sizeof(f_));
		*gretl_data_set->varname = (char *) malloc(sizeof(char));//malloc(f_* size


		::dataset_allocate_varnames(gretl_data_set);
		int err = allocate_Z(gretl_data_set, OPT_NONE);
		/*gretl_data_set->varname[0] = (char *) malloc(sizeof(char));
		strcpy(gretl_data_set->varname[0],  "constant");
		::series_set_label(gretl_data_set, 0, "constant");*/


		for(int i = 0;  i<gretl_data_set->v; ++i){
			gretl_data_set->varname[i] = (char *) malloc(sizeof(char));
			strcpy(gretl_data_set->varname[i],  const_cast<char*>(variable_name[i].c_str()));
			::series_set_label(gretl_data_set, i, variable_name[i].c_str() );
			//::printf("%s\n",gretl_data_set->varname[i]);
		}

		if(gretl_data_set != NULL){
			for(int i = 0; i<nobs;++i)
				gretl_data_set->Z[0][i] = 1.0000;
			int count = 1;
			//for(int j = 0; j<data.size();++j){
			for(auto j : variable_name){
				//data.get_column<double>(j.c_str)[i]
				for(int i = 0; i<nobs;++i){

					if(i > data.get_column<double>(j.c_str()).size()-1 || std::isnan(data.get_column<double>(j.c_str())[i])){
						gretl_data_set->Z[count][i] = NADBL;
					}
					else
						gretl_data_set->Z[count][i] = data.get_column<double>(j.c_str())[i];

					//std::cout<<gretl_data_set->Z[count][i]<<" count "<<count<<" i "<<i<<j<<std::endl;
				}
				++count;
			}
		}
	}


}
template void ToGretl<double>(MyDataFrame &, DATASET *, int , int , std::vector<std::string> , bool );


#endif



/*
 * @brief: performs time delay embedding, inserting the delays right into the data passed in
 *
 * @param data:
 * @param int m: number of embedding dimesnions
 * @param int d: time delay
 * @param bool skip: whether to skip the first value, this is used when when using same data but with different time delays
 *
 */
void EmbedNoReturn(std::multimap<boost::posix_time::ptime, std::vector<long double>> &data ,int m, int d, bool skip){


	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cBegin = data.begin();
	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cEnd = data.end();
	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator iter;
	cEnd--;
	int end = std::abs(m*d)-d+1;
	if(skip == false){
		for( ; cEnd!=cBegin; cEnd--){
			iter = cEnd;
			//cout<<std::distance(cBegin, cEnd)<<endl;

			for(int i =1; i<m; i++){
				std::advance(iter, -1*std::abs(d));

				cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
				//cout<<"cEnd "<<cEnd->second<<" iter "<<iter->second<<endl;
				//cout<<"cEnd "<<cEnd->second<<" iter->second[0] "<<iter->second[0]<<endl;
				//cout<<"cEnd "<<cEnd->second<<" iter->second[1] "<<iter->second[1]<<endl;
			}
			if(std::distance(cBegin, cEnd)<end){
				data.erase(cBegin, cEnd);
				break;
			}
		}
	}
	else{
		for( ; cEnd!=cBegin; cEnd--){
			iter = cEnd;
			//cout<<std::distance(cBegin, cEnd)<<endl;

			for(int i =1; i<m; i++){
				std::advance(iter, -1*std::abs(d));
				//cout<<"cEnd b4 "<<cEnd->second<<" iter "<<iter->second<<endl;
				cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
			}
			cEnd->second.pop_back();

			if(std::distance(cBegin, cEnd)<end){
				data.erase(cBegin, cEnd);
				break;
			}
		}
	}
	//Print(data);
}


void Print(std::multimap<boost::posix_time::ptime, std::vector<long double>> &data){

	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cBegin = data.begin();
	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cEnd = data.end();

	for(;cBegin!=cEnd;cBegin++){
		cout<<cBegin->first<<" ";
		for(std::vector<long double>::iterator vecIter = cBegin->second.begin(); vecIter != cBegin->second.end(); vecIter++)
			cout<<*vecIter<<" ";
		cout<<endl;
	}

}

/*
 *	Combine all string/categorical values in a map then convert that into the vector-> why not just do a set
 *			is there any reason why I would need to  get the key to find its spot in in thevector-> yes when I am saving the data
 *
 *			create map then iterate through it an then incremtn
 *
 *
 *						were int it would be be in vector-> see below about not wanting to use a map
 *	use a map<string, int> were the records it's respective spot in the vector->in short this map is used so I am not constantly iterating over a vector when saving the data
 *	this might be fine for this project but when things need to expand like adding a new state or store, then
 *		the ordering for everything will change because of how it gets inserted into the map/set were as
 *		if it all in a vector i just add it to the end
 *
 *		but the
 *
 *
 *
 *@breif: map
 *
 */

template<typename T>
void OneHotEncodingMap(std::vector<T> &data, std::map<std::string,int> &value_to_vector_idx, std::vector<int> &one_hot_vector){

	std::map<T, int> n_unique;

	Stats::GetUniqueCounts(data,n_unique);

	int idx = one_hot_vector.size();

	for(auto iter:n_unique){
		++idx;
		value_to_vector_idx[std::to_string(iter.first)] = idx;
	}

	one_hot_vector.resize(idx);

}
template void OneHotEncodingMap<double>(std::vector<double> &, std::map<std::string,int> &, std::vector<int> &);
template void OneHotEncodingMap<long double>(std::vector<long double> &, std::map<std::string,int> &, std::vector<int> &);

template<>
void OneHotEncodingMap(std::vector<std::string> &data, std::map<std::string,int> &value_to_vector_idx, std::vector<int> &one_hot_vector){

	std::map<std::string, int> n_unique;

	Stats::GetUniqueCounts(data,n_unique);

	int idx = one_hot_vector.size();

	for(auto iter:n_unique){
		++idx;
		value_to_vector_idx[iter.first] = idx;
	}

	one_hot_vector.resize(idx);

}
template void OneHotEncodingMap<std::string>(std::vector<std::string> &, std::map<std::string,int> &, std::vector<int> &);

//template<>
void CategorizeMap(std::vector<std::string> &data, std::map<std::string,int> &value_to_category_id){

	std::map<std::string, int> n_unique;

	Stats::GetUniqueCounts(data,n_unique);

	int idx = 0;

	for(auto iter:n_unique){
		++idx;
		value_to_category_id[iter.first] = idx;
	}

}
//template void CategorizeMap<std::string>(std::vector<std::string> &, std::map<std::string,int> &);


/*
 * @brief:
 *
 */
template<typename T, typename T1>
void TransFormsInPlace(std::multimap<T, std::vector<std::vector<T1> > > &data, std::vector<int> columns_idx_to_transform, bool value_check, T1 bad_value, std::function<T1 (T1) > func ){


	typename std::multimap<T, std::vector<std::vector<T1> > >::iterator data_iter = data.begin();
	typename std::vector<std::vector<T1> >::iterator outer_vect_iter;
	typename std::vector<T1>::iterator inner_vect_iter;

	for(; data_iter != data.end(); ++data_iter){
		for(int j = 0; j != data_iter->second.size(); ++j){//for(outer_vect_iter= data_iter->second.begin(); outer_vect_iter!= data_iter->second.end(); ++outer_vect_iter){
			for(auto i:columns_idx_to_transform){
				if(value_check == true){
					if( data_iter->second[j][i] != bad_value)
						data_iter->second[j][i] = func(data_iter->second[j][i]);
				}
				else
					data_iter->second[j][i] = func(data_iter->second[j][i]);
			}
		}
	}

}
template void TransFormsInPlace<boost::posix_time::ptime, double>(std::multimap<boost::posix_time::ptime, std::vector<std::vector<double> > > &, std::vector<int>, bool, double, std::function<double (double) >);
template void TransFormsInPlace<boost::posix_time::ptime, long double>(std::multimap<boost::posix_time::ptime, std::vector<std::vector<long double> > > &, std::vector<int>, bool, long double, std::function<long double(long double) >);


/*
 * @brief: iterates through a vector and removes nan's inplace
 *
 */
template<typename T>
void RemoveNANs(std::vector<T> &data){


	std::vector<T> my_data;
	int count_nan=0;

	for(uint i = 0; i<data.size(); ++i){
		if(data[i] != data[i])
			++count_nan;
	}

	if(count_nan>0){
		my_data.resize(data.size() - count_nan);
		int j = 0;
		for(uint i = 0; i<data.size(); ++i){
			if(data[i] == data[i]){
				my_data[j] = data[i];
				++j;
			}
		}
		data.clear();
				data.shrink_to_fit();
				data = my_data;
	}

}
template void RemoveNANs<double>(std::vector<double> &);
template void RemoveNANs<long double>(std::vector<long double> &);

template<>
void RemoveNANs(std::vector<std::string> &data){

	std::string my_nan = "nan";
	std::vector<std::string> my_data;
	int count_nan= std::count(data.begin(),data.end(), my_nan);
	int j = 0;
	my_data.resize(data.size() - count_nan);
	if(count_nan>0){
		for(uint i = 0; i<data.size(); ++i){
			if(data[i].compare(my_nan)!=0){
				my_data[j] = data[i];
				++j;
			}
		}
		data.clear();
		data.shrink_to_fit();
		data = my_data;
	}

}



}




















