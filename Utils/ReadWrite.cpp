/*
 * SaveData.cpp
 *
 *  Created on: May 28, 2020
 *      Author: ryan
 */



#include "ReadWrite.h"



namespace ReadWrite{


//TODO 	1. add check for int then check int every were else
//		2. add check for money
//		3. Check for date??-> would change function to int were 0: not a number, 1: isnumber, 2: isdate
//		4. add check for scientific notation 5.34576E-4
template<typename T>
bool IsNumber(T x){
	    std::string s;
	    x=std::regex_replace(x, std::regex("[ \"]*"), "");
	    std::regex e ("^-?\\d*\\.?\\d+");
	    std::stringstream ss;
	    ss << x;
	    ss >>s;
	    if (std::regex_match (s,e))
	    	return true;
	    else
	    	return false;
}


/*
 * @brief: Reads csv if file has a header then header names and data types are stored in the vector<vector<string> >, also give total number of rows.
 *
 * @tparam T: setting type to either double or long double
 *
 */
template<typename T>
void GetDataTypes(std::string file_name, std::vector<std::vector<std::string> > &names_types,int &n_rows, int &n_columns, bool header){
	std::ifstream my_file(file_name);
	std::string line;
	std::string small_line;

	int column_names_string_iter = 0;
	std::vector<std::string> column_types;
	std::vector<std::string> column_names_string;
	std::vector<double> column_names_double;
	std::vector<std::string> tmp(2);
	int count= 0;
	unsigned total_rows = 0;
	std::string str;


	if(my_file.is_open()){

		my_file.unsetf(std::ios_base::skipws);

		// count the newlines with an algorithm specialized for counting:
		total_rows = std::count(
				std::istream_iterator<char>(my_file),
				std::istream_iterator<char>(),
				'\n');

		my_file.close();
		my_file.open(file_name);
	}

	if (my_file.is_open())
	{


		my_file>>std::ws;

		//read header and get column names, maybe break off into own function
		if(header == true){
			total_rows -= 1;
			std::getline(my_file,line);
			std::stringstream  line_stream(line);
			std::string        cell;
			n_columns = 0;
			while(std::getline(line_stream, cell,',')){

				if(!IsNumber(cell))
					column_names_string.push_back(cell);
				else
					column_names_double.push_back(std::stod(cell));

				++n_columns;
			}
			if(column_names_string.size() > 0 && column_names_double.size() >0){
				std::cout<<"mix of string and doubles in column names "<<std::endl;
				exit(0);
			}
		}

		//get variable types create column vectors
		line=str;//line.clear();
		std::getline(my_file,line);
		std::stringstream  line_stream(line);
		std::string        cell;
		column_types.resize(column_names_string.size());
		names_types.resize(column_names_string.size());
		while(std::getline(line_stream, cell,',')){

			if(!IsNumber(cell)){
				tmp[0] = column_names_string[column_names_string_iter];
				tmp[1] = "std::string";
				names_types[column_names_string_iter] = tmp;
			}
			else{
				if(typeid(T) == typeid(long double)){
					tmp[0] = column_names_string[column_names_string_iter];
					tmp[1] = "long double";
					names_types[column_names_string_iter] = tmp;}
				else{
					tmp[0] = column_names_string[column_names_string_iter];
					tmp[1] = "double";
					names_types[column_names_string_iter] = tmp;
				}
			}
			++column_names_string_iter;
		}
	}
	else
		std::cout<<"File did not open "<<file_name<<std::endl;



}
template void GetDataTypes<double>(std::string , std::vector<std::vector<std::string> > &,int &n_rows, int &n_columns, bool heade);



/*
 * @brief: 	reads a csv file into a DataFrame(https://github.com/hosseinmoein/DataFrame)
 * 			attempts to decide if variable in csv file is a string or double
 *
 * 			MISSING values are given a value of assinged std::nan
 *
 *
 *
 */
template<typename T>
void ReadCSV(std::string file_name, MyDataFrame &df, std::vector<std::string> &column_names_string, bool header){

	std::ifstream my_file(file_name);
	std::string line;
	std::string small_line;

	int column_names_string_iter = 0;
	std::vector<std::string> column_types;
	std::vector<double> column_names_double;
	int n_columns = 0;
	int count= 0;
	unsigned total_rows = 0;
	std::string str;


	if(my_file.is_open()){

		my_file.unsetf(std::ios_base::skipws);

		// count the newlines with an algorithm specialized for counting:
		total_rows = std::count(
				std::istream_iterator<char>(my_file),
				std::istream_iterator<char>(),
				'\n');

		my_file.close();
		my_file.open(file_name);
	}

	if (my_file.is_open())
	{
		my_file>>std::ws;

		//read header and get column names, maybe break off into own function
		if(header == true){
			total_rows -= 1;
			std::getline(my_file,line);
			std::stringstream  line_stream(line);
			std::string        cell;
			n_columns = 0;
			while(std::getline(line_stream, cell,',')){

				if(!IsNumber(cell))
					column_names_string.push_back(cell);
				else
					column_names_double.push_back(std::stod(cell));

				++n_columns;
			}
			if(column_names_string.size() > 0 && column_names_double.size() >0){
				std::cout<<"mix of string and doubles in column names "<<std::endl;
				exit(0);
			}
		}

		//get variable types create column vectors
		line=str;//line.clear();
		std::getline(my_file,line);
		std::stringstream  line_stream(line);
		std::string        cell;
		column_types.resize(column_names_string.size());
		while(std::getline(line_stream, cell,',')){

			if(!IsNumber(cell)){
				df.create_column<std::string>(column_names_string[column_names_string_iter].c_str()).resize(total_rows);
				column_types[column_names_string_iter] = "std::string";
			}
			else{
				df.create_column<T>(column_names_string[column_names_string_iter].c_str()).resize(total_rows);
				if(typeid(T) == typeid(long double))
					column_types[column_names_string_iter] = "long double";
				else
					column_types[column_names_string_iter] = "double";
			}
			++column_names_string_iter;
		}

		my_file.seekg(0, std::ios::beg);
		line = str;
		while(std::getline(my_file,line) ){
			std::stringstream  line_stream(line);
			std::string        cell;

			column_names_string_iter = 0;
			if(header == true && count > 0){
				while(std::getline(line_stream, cell,',')){
					if(cell.size()==0){
						cell = "nan";
						//std::cout<<"blank"<<std::endl;
					}
					if(column_types[column_names_string_iter].compare("std::string") == 0){
						df.get_column<std::string>(column_names_string[column_names_string_iter].c_str())[count-1] = cell;
						//std::cout<<"value "<<df.get_column<std::string>(column_names_string[column_names_string_iter].c_str())[count-1]<<std::endl;
					}
					else{
						if(column_types[column_names_string_iter].compare("double") == 0){
							df.get_column<double>(column_names_string[column_names_string_iter].c_str())[count-1] = std::stod(cell);
							//std::cout<<"value "<<df.get_column<double>(column_names_string[column_names_string_iter].c_str())[count-1]<<std::endl;
						}
						else if(column_types[column_names_string_iter].compare("long double") == 0)
							df.get_column<long double>(column_names_string[column_names_string_iter].c_str())[count-1] = std::stold(cell);
					}
					++column_names_string_iter;
				}
			}
			else if(header == false){
				while(std::getline(line_stream, cell,',')){
					if(cell.size()==0){
						cell = "nan";
						std::cout<<"blank"<<std::endl;
					}
					if(column_types[column_names_string_iter].compare("std::string") == 0)
						df.get_column<std::string>(column_names_string[column_names_string_iter].c_str())[count] = cell;
					else{
						if(column_types[column_names_string_iter].compare("double") == 0)
							df.get_column<double>(column_names_string[column_names_string_iter].c_str())[count] = std::stod(cell);
						else if(column_types[column_names_string_iter].compare("long double") == 0)
							df.get_column<long double>(column_names_string[column_names_string_iter].c_str())[count] = std::stold(cell);
					}
					++column_names_string_iter;
				}
			}
			++count;
		}

	}
	else
		std::cout<<"File did not open "<<file_name<<std::endl;

	my_file.clear();


}
template void ReadCSV<double>(std::string file_name, MyDataFrame &df, std::vector<std::string> &column_names_string, bool header);
//template void ReadCSV<float>(std::string file_name, MyDataFrame &df, std::vector<std::string> &column_names_string, bool header);

/*
 * @brief: 	reads a csv file into a DataFrame(https://github.com/hosseinmoein/DataFrame)
 * 			attempts to decide if variable in csv file is a string or double
 *
 * 			MISSING values are given a value of assinged std::nan
 *
 *
 *
 */
template<typename T>
void ReadCSV(std::string file_name, std::vector<T> &df, std::vector<std::string> &column_names_string, bool header){

	std::ifstream my_file(file_name);
	std::string line;
	std::string small_line;

	int column_names_string_iter = 0;
	std::vector<std::string> column_types;
	std::vector<double> column_names_double;
	int n_columns = 0;
	int count= 0;
	unsigned total_rows = 0;
	std::string str;
	std::string v="V";//name columns when column_names_string is empty

	if(my_file.is_open()){

		my_file.unsetf(std::ios_base::skipws);

		// count the newlines with an algorithm specialized for counting:
		total_rows = std::count(
				std::istream_iterator<char>(my_file),
				std::istream_iterator<char>(),
				'\n');

		my_file.close();
		my_file.open(file_name);
	}

	if (my_file.is_open())
	{
		my_file>>std::ws;

		//read header and get column names, maybe break off into own function
		if(header == true){
			total_rows -= 1;
			std::getline(my_file,line);
			std::stringstream  line_stream(line);
			std::string        cell;
			n_columns = 0;
			while(std::getline(line_stream, cell,',')){

				if(!IsNumber(cell))
					column_names_string.push_back(cell);
				else
					column_names_double.push_back(std::stod(cell));

				++n_columns;
			}
			if(column_names_string.size() > 0 && column_names_double.size() >0){
				std::cout<<"mix of string and doubles in column names "<<std::endl;
				exit(0);
			}

			column_types.resize(column_names_string.size());
		}
		else{
			std::getline(my_file,line);
			std::stringstream  line_stream(line);
			std::string        cell;
			n_columns = 0;
			while(std::getline(line_stream, cell,',')){

				v = "V"+std::to_string(n_columns);
				column_names_string.push_back(v);
				++n_columns;
			}

			column_types.resize(n_columns);
			//column_names.resize(n_columns)
		}

		//get variable types create column vectors
		line=str;//line.clear();
		std::getline(my_file,line);
		std::stringstream  line_stream(line);
		std::string        cell;
		while(std::getline(line_stream, cell,',')){

			if(!IsNumber(cell)){
				//df.create_column<std::string>(column_names_string[column_names_string_iter].c_str()).resize(total_rows);
				column_types[column_names_string_iter] = "std::string";
			}
			else{
				//df.create_column<T>(column_names_string[column_names_string_iter].c_str()).resize(total_rows);
				if(typeid(T) == typeid(long double))
					column_types[column_names_string_iter] = "long double";
				else
					column_types[column_names_string_iter] = "double";
			}
			++column_names_string_iter;
		}

		df.resize(total_rows*column_types.size());
		my_file.seekg(0, std::ios::beg);
		line = str;
		while(std::getline(my_file,line) ){
			std::stringstream  line_stream(line);
			std::string        cell;

			column_names_string_iter = 0;
			if(header == true && count > 0){
				while(std::getline(line_stream, cell,',')){
					if(cell.size()==0){
						cell = "nan";
						//std::cout<<"blank"<<std::endl;
					}
					if(column_types[column_names_string_iter].compare("std::string") == 0){
						continue;//df[count] = cell;
						//std::cout<<"value "<<df.get_column<std::string>(column_names_string[column_names_string_iter].c_str())[count-1]<<std::endl;
					}
					else{
						if(column_types[column_names_string_iter].compare("double") == 0){
							df[count] = std::stod(cell);
							//std::cout<<"value "<<df.get_column<double>(column_names_string[column_names_string_iter].c_str())[count-1]<<std::endl;
						}
						else if(column_types[column_names_string_iter].compare("long double") == 0)
							df[count] = std::stold(cell);
					}
					++column_names_string_iter;
					++count;
				}
			}
			else if(header == false){
				while(std::getline(line_stream, cell,',')){
					if(cell.size()==0){
						cell = "nan";
						std::cout<<"blank"<<std::endl;
					}
					if(column_types[column_names_string_iter].compare("std::string") == 0)
						continue;//df[count] = cell;
					else{
						if(column_types[column_names_string_iter].compare("double") == 0)
							df[count] = std::stod(cell);
						else if(column_types[column_names_string_iter].compare("long double") == 0)
							df[count] = std::stold(cell);
					}
					++column_names_string_iter;
					++count;
				}
			}

		}

	}
	else
		std::cout<<"File did not open "<<file_name<<std::endl;

	my_file.clear();


}
template void ReadCSV<double>(std::string file_name, std::vector<double> &df, std::vector<std::string> &column_names_string, bool header);
template void ReadCSV<float>(std::string file_name, std::vector<float> &df, std::vector<std::string> &column_names_string, bool header);
//template void ReadCSV<st>(std::string file_name, std::vector<std::string> &df, std::vector<std::string> &column_names_string, bool header);




/*
 * @brief: 	opens a file puts data into a map<T1, std:;vector<T2>> skips_columns in skip_columns
 * 			Missing values are assinged std::nan
 *
 *@param string file_name: name and path of csv file to open
 *@tparam map<T1,vector<T2> > the_data: container to put data into
 *@param bool header: true to skip header false = no header
 *@param set<int> skip_columns: columns numbers to skip
 *
 */
template<typename T1, typename T2>
void ReadCSV(std::string file_name, std::map<T1, std::vector<T2 > > &the_data, bool header, std::set<int> &skip_columns){


	std::ifstream my_file(file_name);
	std::string line;
	int count = 0;
	std::time_t my_time;
	boost::posix_time::ptime my_ptime;
	long double order_id = 00000000.000000000000;
	int n_columns=0;
	int tmp_count = 0;
	int vector_size = 0;
	std::vector<T2> tmp;
	std::vector<T2> clear_tmp;

	T1 row_id;

	//skip header
	if(header){
		std::getline(my_file, line);
		std::stringstream  lineStream(line);
		std::string        cell;
		while(std::getline(lineStream,cell,','))
			++n_columns;
	}

	vector_size = n_columns - skip_columns.size();
	tmp.resize(vector_size);
	while(std::getline(my_file, line)){
		std::stringstream  lineStream(line);
		std::string        cell;

		count = 0;
		tmp_count = 0;
		while(std::getline(lineStream,cell,','))
		{
			if(cell.size()==0){
				cell = "nan";
				std::cout<<"blank"<<std::endl;
			}
			if(count == 0){
				row_id = cell;
			}
			else if(skip_columns.size() > 0 && skip_columns.find(count)==skip_columns.end()){
				if(typeid(T2) == typeid(double))
					tmp[tmp_count] = std::stod(cell);
				else if(typeid(T2) == typeid(long double))
					tmp[tmp_count] = std::stold(cell);

				++tmp_count;
			}
			else if(skip_columns.size() == 0){
				if(typeid(T2) == typeid(double))
					tmp[tmp_count] = std::stod(cell);
				else if(typeid(T2) == typeid(long double))
					tmp[tmp_count] = std::stold(cell);

				++tmp_count;
			}

			++count;

		}
		the_data.insert(std::pair<T1, std::vector<T2> >(row_id, tmp));
		tmp.clear();
		tmp = clear_tmp;
		tmp.resize(vector_size);

	}



}
template void ReadCSV<std::string, double>(std::string , std::map<std::string, std::vector<double > > &, bool, std::set<int> &);



/*
 * @brief: reads the file and puts it into a string
 *
 */
void ReadFile(std::string file_name, std::string &data){

	std::ifstream my_file(file_name);

	if (my_file.is_open())
	{
		std::string contents;
		my_file.seekg(0, std::ios::end);
		data.resize(my_file.tellg());
		my_file.seekg(0, std::ios::beg);
		my_file.read(&data[0], data.size());
		my_file.close();
	}
	else
		std::cout<<"File did not open "<<file_name<<std::endl;

}

void Unzip(std::string file_path_name, std::string &data){

	std::cout<<"Unzip start"<<std::endl;



	boost::process::ipstream sstream;
	std::vector<std::string> args;
	args.push_back("-p");
	args.push_back(file_path_name);
	boost::process::child c("/usr/bin/unzip",args, boost::process::std_out > sstream);



	std::stringstream please_work;
	boost::iostreams::filtering_streambuf<boost::iostreams::input> def;
	def.push(sstream);
	boost::iostreams::copy(def, please_work);

	data = please_work.str();

	c.wait();



	std::cout<<"Unzip end"<<std::endl;

}

/*
 * @brief: reads a zip file and puts the data into a string
 * @ISSUE: I don't think is true -> seems to not realse memory, some observations show that it is no worse than Unzip()
 *
 */
void ReadCSVZIP(std::string file_name, std::string &data){

		std::string path = file_name;//"/home/ryan/DI/DITestData/OFER_VDA_BMF_20170814.zip";
		std::string all_data;

		std::vector<std::string> tmp_vect;
		std::vector<std::vector<std::string> > fnl_data;

		//Open the ZIP archive
		int err = 0;
		//std::vector<std::string> filename;
		//filename=iterdir(file_path);

		zip *z = zip_open(path.c_str(), 0, &err);

		//Search for the file of given name
		//std::string file_name=file_nameTemp;//"OFER_VDA_BMF_20170814.TXT";
		//const char *name = file_name.c_str();
		struct zip_stat st;
		zip_stat_init(&st);
		//zip_stat(z, file_name.c_str(), 0, &st);

		//in version 1.5.1 cout<<zip_libzip_version()<<endl;

		if(z!=NULL){
			const char *n=zip_get_name(z,0,0);
			zip_stat(z, n, 0, &st);

			//Alloc memory for its uncompressed contents
			char *contents = new char[st.size];

			int *what=0;
			struct zip *zp;
			zp = zip_fdopen(ZIP_RDONLY,1,what);

			//Read the compressed file
			//zip_file *f = zip_fopen(z, file_name.c_str(), 0);
			zip_file *f = zip_fopen(z, n, 0);

			if(zip_fread(f, contents, st.size)==-1){
				std::cout<<"zip_fread did not read anything "<<std::endl;


			}
			else{
				//std::string fuck = zip_strerror(z);
				std::string fuck = zip_file_strerror(f);

				if(contents !=NULL){
					data = std::move(contents);//myString;
				}

				zip_fclose(f);
				f=NULL;

				//And close the archive
				zip_close(z);
				z=NULL;
			}

			//Do something with the contents
			//delete allocated memory
			if(contents !=NULL){
				//contents = NULL;
				delete[] contents;
			}


		}

}





/*
 * @brief
 *
 *NOTE: to_string precision is ~ 6 place to left of deceimal, if need to save hight precision will need to do something else
 *
 *
 * if not working need to add SaveData<double, const std::map<std::string, std::multimap<boost::gregorian::date, std::vector<double> > > >(path, name , data)
 */

template<typename T, typename T1>
void SaveData(std::string file_path, std::string name, const  T1 &data){

	boost::gregorian::date date;


	std::string time;
	int err = 0;
	for(auto mapIt: data){
		zip_source *s;//zip_source_t
		std::string path_ = file_path+name+mapIt.first+".zip";
		std::string filename = file_path+name+mapIt.first+".csv";
		zip *archive = zip_open(path_.c_str(), ZIP_CREATE, &err);
		std::string my_name = "name_"+mapIt.first+".csv";
		std::string fnlData;
		for(auto i:mapIt.second){
			boost::posix_time::ptime new_time(i.first);
			time=ReadWrite::DateTimeToString(new_time);//boost::gregorian::to_iso_string(i.first);//std::to_string(i.first);
			fnlData.append(time+',');
			for(auto j:i.second)
				fnlData.append(std::to_string(j)+',');
			fnlData.append("\n");
		}

		if ((s=zip_source_buffer(archive, fnlData.c_str(),fnlData.size(), 0)) == NULL ||zip_add(archive,my_name.c_str(), s)>0) {
			zip_source_free(s);
			printf("error adding file: %s\n", zip_strerror(archive));
		}
		zip_close(archive);
	}


}
template void SaveData<double, const std::map<std::string, std::multimap<boost::gregorian::date, std::vector<long double> > > >(std::string path, std::string , const std::map<std::string, std::multimap<boost::gregorian::date, std::vector<long double> > > &);
template void SaveData<double, const std::map<std::string, std::multimap<boost::gregorian::date, std::vector<double> > > >(std::string , std::string , const std::map<std::string, std::multimap<boost::gregorian::date, std::vector<double> > > &);
template void SaveData<double, const std::map<std::string, std::map<boost::gregorian::date, std::vector<double> > > >(std::string , std::string , const std::map<std::string, std::map<boost::gregorian::date, std::vector<double> > > &);


template<typename T, typename T1 >
void SaveDataVector(std::string file_path, std::string name, const T1 &data){

	boost::gregorian::date date;


	std::string time;
	int err = 0;
	zip_source *s;//zip_source_t
	std::string path_ = file_path+name+".zip";
	std::string filename = file_path+name+".csv";
	zip *archive = zip_open(path_.c_str(), ZIP_CREATE, &err);
	std::string my_name = "name_"+name+".csv";
	std::string fnlData;
	for(auto mapIt: data){


		for(auto i:mapIt){
			//boost::posix_time::ptime new_time(i.first);
			//time=ReadWrite::DateTimeToString(new_time);//boost::gregorian::to_iso_string(i.first);//std::to_string(i.first);
			//fnlData.append(time+',');
			//for(auto j:i.second)
			fnlData.append(std::to_string(i)+',');

		}
		fnlData.append("\n");


	}
	if ((s=zip_source_buffer(archive, fnlData.c_str(),fnlData.size(), 0)) == NULL ||zip_add(archive,my_name.c_str(), s)>0) {
		zip_source_free(s);
		printf("error adding file: %s\n", zip_strerror(archive));
	}
	zip_close(archive);


}
template void SaveDataVector<double, const std::vector<std::vector<double> > >(std::string path, std::string , const std::vector<std::vector<double> > &);






//TODO: implement skip_columns set
/*
 *
 * @param int n_levels: number of levels to get 1 = one level, 2 = 2 levels, ... , n=max levels in file
 * @param int level: if a specific level is wanted,default = -1 = all levels,  ex: 1 = top, 2 = 2nd level ...
 */
template<typename T, typename T1>
void ParseString(std::string &data, std::multimap<T, std::vector<std::vector<T1> > > &final_data, int column_idx_to_key,  int outer_vect_size, int inner_vect_size, std::set<int> skip_columns){


	std::istringstream stream(data);
	std::string line;
	int outer_count = 0;
	int inner_count = 0;
	int skip_columns_size = skip_columns.size();
	std::time_t my_time;
	T my_ptime;

	T1 value = 00000000.000000000000;
	std::vector<std::vector<T1> > outer_vect(outer_vect_size);

	while(std::getline(stream, line)){
		std::vector<T1> inner_vect(inner_vect_size);
		std::stringstream  lineStream(line);
		std::string        cell;

		inner_count = 0;
		while(std::getline(lineStream,cell,','))
		{
			if(inner_count != column_idx_to_key ){

				if(skip_columns_size == 0){
					if(inner_count < inner_vect.size()){
						value = std::stold(cell);
						inner_vect[inner_count-1] = value;
					}
				}
				else if(skip_columns.find(inner_count) != skip_columns.end()){
					//to be implemented
				}

			}
			else{
				my_time = std::stold(cell);
				my_ptime = my_time;//change later // boost::posix_time::from_time_t(my_time/1000000000)+boost::posix_time::microsec(my_time % 1000);
			}
			++inner_count;
		}

		if(outer_count < outer_vect_size){
			outer_vect[outer_count] = inner_vect;
			++outer_count;

		}
		else{
			final_data.insert(std::pair<T,std::vector<std::vector<T1> > >(my_ptime, outer_vect));
			outer_count = 0;
		}

	}


}
template void ParseString<long double, long double>(std::string &, std::multimap<long double, std::vector<std::vector<long double> > > &, int ,  int, int , std::set<int> );


//TODO: implement skip_columns set
/*
 *
 * @param int n_levels: number of levels to get 1 = one level, 2 = 2 levels, ... , n=max levels in file
 * @param int level: if a specific level is wanted,default = -1 = all levels,  ex: 1 = top, 2 = 2nd level ...
 */
template<typename T1>
void ParseString(std::string &data, std::multimap<boost::posix_time::ptime, std::vector<std::vector<T1> > > &final_data, int column_idx_to_key,  int outer_vect_size, int inner_vect_size, std::set<int> skip_columns){


	std::cout<<"ParseString data size "<<data.size()<<std::endl;
	std::string fuck = std::move(data);
	std::stringstream stream(fuck);

	std::string line;
	int outer_count = 0;
	int inner_count = 0;
	int skip_columns_size = skip_columns.size();
	std::time_t my_time;
	boost::posix_time::ptime my_ptime;
	boost::posix_time::ptime my_ptime1;
	int count = 0;
	bool bad_data = false;

	long double value = 00000000.000000000000;
	std::vector<std::vector<T1 > > outer_vect(outer_vect_size);

	while(std::getline(stream, line)){
		std::vector<T1> inner_vect(inner_vect_size);
		std::stringstream  lineStream(line);
		std::string        cell;

		inner_count = 0;
		bad_data = false;
		//std::cout<<line<<std::endl;
		while(std::getline(lineStream,cell,','))
		{
			if(inner_count != column_idx_to_key ){

				if(skip_columns_size == 0){
					if(inner_count < inner_vect.size()){
						value = std::stold(cell);
						inner_vect[inner_count-1] = value;
					}
				}
				else if(skip_columns.find(inner_count) != skip_columns.end()){
					//to be implemented
				}

			}
			else{
				if(ReadWrite::IsNumber(cell)){
					my_time = std::stold(cell);
					my_ptime = boost::posix_time::from_time_t(my_time / 1000000000)+boost::posix_time::microseconds(boost::posix_time::microsec(my_time / 1000).fractional_seconds());
					//my_ptime += boost::posix_time::microsec(my_time / 1000).fractional_seconds();//+ boost::posix_time::microsec(my_time % 1000);
					//std::cout<<my_ptime<<" "<< boost::posix_time::microsec(my_time / 1000)<<" "<<boost::posix_time::microsec(my_time / 1000).fractional_seconds()<<" "<<boost::posix_time::microseconds(boost::posix_time::microsec(my_time / 1000).fractional_seconds())<<std::endl;
				}
				else{
					inner_vect.clear();
					//outer_vect.clear();
					lineStream.flush();
					outer_count = 0;
					bad_data = true;
					break;

				}
			}
			++inner_count;
		}

		if(bad_data == false  && outer_count < outer_vect_size){
			outer_vect[outer_count] = inner_vect;
			++outer_count;

		}
		if(bad_data == false && outer_count == outer_vect_size){
			final_data.insert(final_data.end(), std::pair<boost::posix_time::ptime,std::vector<std::vector<T1> > >(my_ptime, outer_vect));
			outer_count = 0;
			++count;
		}

	}


	std::cout<<"ParseString end"<<std::endl;

}
template void ParseString<double>(std::string &, std::multimap<boost::posix_time::ptime, std::vector<std::vector<double> > > &, int ,  int, int , std::set<int> );
template void ParseString<long double>(std::string &, std::multimap<boost::posix_time::ptime, std::vector<std::vector<long double> > > &, int ,  int, int , std::set<int> );




std::string DateToString(boost::posix_time::ptime datetime){

	std::ostringstream convert;
	convert<<datetime.date().year();
	if(datetime.date().month().as_number()<10)
		convert<<"0"<<datetime.date().month().as_number();
	else
		convert<<datetime.date().month().as_number();

	if(datetime.date().day()>=10)
		convert<<datetime.date().day();
	else
		convert<<"0"<<datetime.date().day();

	//convert<<" ";
	//convert<<datetime

	std::string fuck = convert.str();

	return convert.str();



}

/*
 *  @brief iterates through a directory and searches for all files that contain one of the search
 *  	phrases in the set find_in_name
 */
std::vector<std::string> iterdir(std::string file_path, std::set<std::string> &find_in_name){
	DIR *dirp = opendir(file_path.c_str());

	std::vector<std::string> filename;
	std::string name;
	std::string trade="NEG";

	while(dirp){
		errno=0;
		dirent *dp = readdir(dirp);
		if(dp!=NULL){
			name = dp->d_name;
			if(find_in_name.find(name.substr(0, find_in_name.begin()->size())) != find_in_name.end())
				filename.push_back(file_path + name);
			else{
				auto iter = find_in_name.begin();
				++iter;//iter once since we already checked the first element
				for(; iter != find_in_name.end(); ++iter){
					if(find_in_name.find(name.substr(0, iter->size())) != find_in_name.end())
						filename.push_back(file_path + name);
				}

			}
		}
		else
			if(errno==0){
				closedir(dirp);
				break;
			}
	}

	return filename;

}

/*
 *  @brief iterates through a directory and searches for all files that contain one of the search
 *  	phrases in the set find_in_name
 */
std::vector<std::string> iterdir(std::string file_path, std::string &find_in_name){
	DIR *dirp = opendir(file_path.c_str());

	std::vector<std::string> filename;
	std::string name;
	std::string trade="NEG";

	while(dirp){
		errno=0;
		dirent *dp = readdir(dirp);
		if(dp!=NULL){
			name = dp->d_name;
			if(name.find(find_in_name) != std::string::npos)
				filename.push_back(file_path + name);
			/*else{
				auto iter = find_in_name.begin();
				++iter;//iter once since we already checked the first element
				for(; iter != find_in_name.end(); ++iter){
					if(find_in_name.find(name.substr(0, iter->size())) != find_in_name.end())
						filename.push_back(file_path + name);
				}

			}*/
		}
		else
			if(errno==0){
				closedir(dirp);
				break;
			}
	}

	return filename;

}


/*
 *  @brief iterates through a directory and searches for all files that contain one of the search
 *  	phrases in the set find_in_name also appends a string to the end of the name
 */
std::vector<std::string> iterdir(std::string file_path, std::set<std::string> &find_in_name, std::string to_append){
	DIR *dirp = opendir(file_path.c_str());

	std::vector<std::string> filename;
	std::string name;
	std::string trade="NEG";

	while(dirp){
		errno=0;
		dirent *dp = readdir(dirp);
		if(dp!=NULL){
			name = dp->d_name;
			if(find_in_name.find(name.substr(0, find_in_name.begin()->size())) != find_in_name.end())
				filename.push_back(file_path + name + to_append);
			else{
				auto iter = find_in_name.begin();
				++iter;//iter once since we already checked the first element
				for(; iter != find_in_name.end(); ++iter){
					if(find_in_name.find(name.substr(0, iter->size())) != find_in_name.end())
						filename.push_back(file_path + name+to_append);
				}

			}
		}
		else
			if(errno==0){
				closedir(dirp);
				break;
			}
	}

	return filename;

}


boost::gregorian::date DateFromString(std::string time){

	int y = std::stoi(time.substr(0,4));
	int m = std::stoi(time.substr(4,2));
	int d = std::stoi(time.substr(6,2));

	return boost::gregorian::date(y,m,d);

}

template<typename T>
boost::posix_time::ptime EpochToPtimeNanoSeconds(const T &epoch){

	std::time_t curent_time = (long double)epoch;
	return boost::posix_time::from_time_t(curent_time / 1000000000)+boost::posix_time::microseconds(boost::posix_time::microsec(curent_time / 1000).fractional_seconds());

}
template boost::posix_time::ptime EpochToPtimeNanoSeconds<uint64_t>(const uint64_t &);
template boost::posix_time::ptime EpochToPtimeNanoSeconds<long long>(const long long &);

std::string DateTimeToString(boost::posix_time::ptime datetime){

	std::ostringstream convert;
	convert<<datetime.date().year();
	if(datetime.date().month().as_number()<10)
		convert<<"0"<<datetime.date().month().as_number();
	else
		convert<<datetime.date().month().as_number();

	if(datetime.date().day()>=10)
		convert<<datetime.date().day();
	else
		convert<<"0"<<datetime.date().day();

	convert<<" ";
	convert<<datetime.time_of_day();

	return convert.str();



}




}
