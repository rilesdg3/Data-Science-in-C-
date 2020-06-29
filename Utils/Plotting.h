/*
 * Plotting.h
 *
 *  Created on: Jun 26, 2020
 *      Author: ryan
 */

#ifndef PLOTTING_H_
#define PLOTTING_H_


#include <mgl2/mgl.h>
#include <algorithm>






/*
 * @brief: helper function building a bar graph using mathgl
 * 			copies copies data to a mglData object
 *
 * 	@param:
 *
 */
template<typename T, typename T1>
void MGLBars(std::vector<T> &data, std::vector<T1> &variable_names, std::string title, std::string save_as)
{
	mglGraph *gr = new mglGraph;
	int size = data.size();
	mglData ys(data.size());
	auto result = std::minmax_element (data.begin(),data.end());
	int max = *result.second;
	std::string tmp="";
	for(int i = 0; i <size; ++i)
		ys[i] = (int)data[i];

	gr->Title(title.c_str());
	gr->Box();
	gr->SetTickTempl('x'," ");

	gr->SetRanges(0,size,0,max);
	for(uint i = 0; i<variable_names.size(); ++i){
		tmp = std::to_string(variable_names[i]);
		gr->AddTick('x',(i+0.5),tmp.c_str());
	}

	gr->Axis("a");
	gr->Bars(ys);

	gr->WritePNG(save_as.c_str(),"",false);
	delete gr;
}

template<typename T>
void MGLBars(std::vector<T> &data, std::vector<std::string> &variable_names, std::string title, std::string save_as)
{
	mglGraph *gr = new mglGraph;
	int size = data.size();
	mglData ys(data.size());
	auto result = std::minmax_element (data.begin(),data.end());
	int max = *result.second;
	std::string tmp="";
	for(int i = 0; i <size; ++i)
		ys[i] = (int)data[i];

	gr->Title(title.c_str());
	gr->Box();
	gr->SetTickTempl('x'," ");

	gr->SetRanges(0,size,0,max);
		for(uint i = 0; i<variable_names.size(); ++i)
			gr->AddTick('x',(i+0.5),variable_names[i].c_str());


	gr->Axis("a");
	gr->Bars(ys);

	gr->WritePNG(save_as.c_str(),"",false);
	delete gr;
}






#endif /* PLOTTING_H_ */
