/*
 * SaveData.cpp
 *
 *  Created on: May 28, 2020
 *      Author: ryan
 */



#include "ReadWrite.h"


namespace ReadWrite{


//TODO add check for money
template<typename T>
bool IsNumber(T x){
	    std::string s;
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
 * @brief: Reads csv if file has a header then header names and data types are stroed in the map, also give total number of rows.
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
						std::cout<<"blank"<<std::endl;
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


}
template void ReadCSV<double>(std::string file_name, MyDataFrame &df, std::vector<std::string> &column_names_string, bool header);


void ReadFile(std::string file_name, std::vector<std::vector<std::string> > &the_data, bool header, int columns){

/*	int lineIter = 0;
		std::vector<long double> tmp;
		boost::posix_time::ptime bidTime;
		std::ifstream  data(file_name);

		std::string line;


		while(std::getline(data,line))
		{
			std::stringstream  lineStream(line);
			std::string        cell;
			while(std::getline(lineStream,cell,','))
			{
				if(lineIter!=0){
					tmp.push_back(stold(cell));
				}
				else{
					bidTime= DateTimeFromString(cell);
					//cout<<bidTime<<endl;
					lineIter++;
				}
			}

			bboPlus.insert(std::pair<boost::posix_time::ptime, std::vector<long double> >
			(bidTime, tmp));
			lineIter =0;
			tmp.clear();
		}*/



}

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


/*
 * @brief: reads a zip file and puts the data into a string
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
					data = contents;//myString;
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
 */
void SaveData(std::string file_path, std::string name, const std::map<std::string, std::map<boost::gregorian::date, std::vector<long double> > > &data){

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
