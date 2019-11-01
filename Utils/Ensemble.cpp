/*
 * Ensemble.cpp
 *
 *  Created on: Aug 9, 2019
 *      Author: ryan
 */


#include <Ensemble.h>





//#define MAP 1



std::set<int> TRTEVA_set;
std::set<int> TETRVA_set;
std::set<int> VATRTE_set;
void CreateSets(int n_TRTEVA, int n_TETRVA, int n_VATRTE){
	for(int i = 0; i<n_TRTEVA; ++i)
		TRTEVA_set.insert(i);
	for(int i = n_TRTEVA; i<n_TETRVA; ++i)
		TETRVA_set.insert(i);
	for(int i = n_TETRVA; i<n_VATRTE; ++i)
		VATRTE_set.insert(i);
}

/*
 * @brief decides which combinations of models can be used in the ensemble
 *
 * @param: vector<int> combos vector containing the model number
 * @param: int n_models_per_set minimum number of models to use from each group
 * 			if ensemble size is 3 and n_models_per_set = 1 then it will use 1 model from TRTEVA 1 from TETRVA and 1 From VATRTE
 * 			if ensemble size is 5 and n_models_per_set = 1 then will use at least 1 models from from each group 2 models from one group
 */
bool AllowCombo(vector<int> combos, int n_models_per_set = 3){

	vector<int> intersect(10);
	vector<int> intersect1(10);
	vector<int> intersect2(10);
	std::vector<int>::iterator it;
	bool in_TRTEVA = false;
	bool in_TETRVA = false;
	bool in_VATRTE= false;

	//cout<<"AllowCombo combos "<<combos<<endl;

	it = std::set_intersection(combos.begin(), combos.end(), TRTEVA_set.begin(), TRTEVA_set.end(),intersect.begin());
	if(std::distance(intersect.begin(), it)>=n_models_per_set)
		in_TRTEVA=true;

	it = std::set_intersection(combos.begin(), combos.end(), TETRVA_set.begin(), TETRVA_set.end(),intersect1.begin());
	if(std::distance(intersect1.begin(), it)>=n_models_per_set)
		in_TETRVA=true;

	it = std::set_intersection(combos.begin(), combos.end(), VATRTE_set.begin(), VATRTE_set.end(),intersect2.begin());
	if(std::distance(intersect2.begin(), it)>=n_models_per_set)
		in_VATRTE=true;

	if(in_TRTEVA==true && in_TETRVA ==true && in_VATRTE == true)
		return true;
	else
		return false;

}

vector<vector<int> > Combinations(vector<int> iterable, int r){
	//# combinations('ABCD', 2) --> AB AC AD BC BD CD
	//# combinations(range(4), 3) --> 012 013 023 123
	CreateSets();

	int max = *std::max(iterable.begin(),iterable.begin()+iterable.size()-1);
	//auto pool = iterable;
	int n = iterable.size();//len(pool)
	vector<int >ranges;
	vector<vector<int> > all_combinations;
	vector<int > tmp(r);

	if (r > n)
		return all_combinations;
	//indices = list(range(r))
	vector<int> indices;
	for(int it = 0; it<r; it++)
		indices.push_back(it);

	//yield tuple(pool[i] for i in indices)
	for(auto it = 0; it < indices.size(); ++it)
		tmp[it] = indices[it];//pool[it] = indices[it];
	if(*std::max(tmp.begin(),tmp.begin()+tmp.size()-1)<=max && AllowCombo(tmp))
		all_combinations.push_back(tmp);

	for(int i =0; i<r;++i)
		ranges.push_back(i);

	std::reverse(ranges.begin(),ranges.end());
	int i= ranges[0];
	while(indices[r-1]<n){
		for(; i>ranges[ranges.size()-1]; --i){//in reversed(range(r)):
			if(indices[i] != i + n - r)
				break;
		}
		indices[i] += 1;

		for(int j = i+1; j<r;++j)
			indices[j] = indices[j-1] + 1;

		for(auto it = 0; it < indices.size(); ++it)
			tmp[it] = indices[it];//pool[it] = indices[it];
		//cout<<"tmp "<<tmp<<endl;
		//cout<<"indices "<<indices<<endl;
		if(*std::max(tmp.begin(),tmp.begin()+tmp.size()-1)<=max && AllowCombo(tmp))
			all_combinations.push_back(tmp);
		i= ranges[0];
	}
	//cout<<"tmp "<<tmp<<endl;
	//cout<<"all_combinations "<<all_combinations[0]<<endl;
	return all_combinations;
}

/*
 * @brief compares stat to the key in multimap to decided if the ensemble is good enough for further review
 * @param int stat: the comparision statistic for inclusion
 * @param multimap potential_ensembles: the map of ensembles that have the potential to be the final ensemble
 * 			were int is the stat we are looking at and map<vector, StatsTuple> is the models in the ensemble and StatTuple is the stats
 * 			of the ensemble
 *
 */
void PotentialEnesmbleCheck(int stat, std::map<int, std::map<vector<int>, StatsTuple > > &potential_ensembles, vector<int> ensemble, StatsTuple ensemble_stats){


	if(stat > potential_ensembles.begin()->first){

		if(potential_ensembles.find(stat)==potential_ensembles.end()){
			std::map<vector<int>, StatsTuple > stat_map;
			std::map<vector<int>, StatsTuple>::iterator stat_map_it;
			stat_map_it = potential_ensembles.begin()->second.begin();
			potential_ensembles.begin()->second.erase(stat_map_it);

			//std::multimap<int, std::map<vector<int>, StatsTuple > > pe_it;
			stat_map[ensemble] = ensemble_stats;
			potential_ensembles.erase(potential_ensembles.begin()->first);

			//potential_ensembles.insert(std::pair<int, std::map<vector<int>, StatsTuple > >(stat, stat_map));
			potential_ensembles[stat] = stat_map;
		}
		else
			potential_ensembles[stat][ensemble] = ensemble_stats;


	}


}


std::set<std::vector<int> > PotentialEnsemblesToSet(std::map<int, std::map<vector<int>, StatsTuple > > &ensemble){

	std::set<std::vector<int> > ensemble_set;
	std::set<std::vector<int> > ensemble_set1;

	std::map<vector<int>, StatsTuple > ensemble_stats;

	for(auto stat_it = ensemble.begin(); stat_it != ensemble.end(); ++stat_it){
		for(auto pe_it = stat_it->second.begin(); pe_it != stat_it->second.end(); ++pe_it){
			ensemble_set.insert(pe_it->first);
			ensemble_stats[pe_it->first] = pe_it->second;
		}

	}

	return ensemble_set;

}

void FindIntersectingEnsemblesSets(std::map<int, std::map<vector<int>, StatsTuple > > &ensemble, std::map<int, std::map<vector<int>, StatsTuple > > &ensemble1,
		std::map<int, std::map<vector<int>, StatsTuple > > &ensemble2){

	//std::map<int, std::map<vector<int>, StatsTuple > >::iterator stat_it;
	std::map<vector<int>, StatsTuple>::iterator ensemble_it;

	std::set<std::vector<int> > ensemble_set = PotentialEnsemblesToSet(ensemble);
	std::set<std::vector<int> > ensemble_set1 = PotentialEnsemblesToSet(ensemble1);

	std::set<std::vector<int > > ensemble_intersection;

	std::set<std::vector<int> >::iterator set_it;
	for(auto it = ensemble_set.begin(); it != ensemble_set.end(); ++it){
		set_it = ensemble_set1.find(*it);
		if(set_it != ensemble_set1.end())
			cout<<"We got a winner "<<*set_it<<endl;
	}


	cout<<"break "<<endl;

	//auto intersect = std::set_intersection(ensemble_set.begin(), ensemble_set.end(), ensemble_set1.begin(), ensemble_set1.end(), ensemble_intersection.begin());

}

std::map<vector<int>, StatsTuple > PotentialEnsemblesToMap(std::map<int, std::map<vector<int>, StatsTuple > > &ensemble){

	std::set<std::vector<int> > ensemble_set;
	std::set<std::vector<int> > ensemble_set1;

	std::map<vector<int>, StatsTuple > ensemble_stats;

	for(auto stat_it = ensemble.begin(); stat_it != ensemble.end(); ++stat_it){
		for(auto pe_it = stat_it->second.begin(); pe_it != stat_it->second.end(); ++pe_it){
			ensemble_set.insert(pe_it->first);
			ensemble_stats[pe_it->first] = pe_it->second;
		}

	}

	return ensemble_stats;

}

void FindIntersectingEnsemblesMap(std::map<int, std::map<vector<int>, StatsTuple > > &ensemble, std::map<int, std::map<vector<int>, StatsTuple > > &ensemble1,
		std::map<int, std::map<vector<int>, StatsTuple > > &ensemble2){

	//std::map<int, std::map<vector<int>, StatsTuple > >::iterator stat_it;
	std::map<vector<int>, StatsTuple>::iterator ensemble_it;

	std::map<vector<int>, StatsTuple > ensemble_set = PotentialEnsemblesToMap(ensemble);
	std::map<vector<int>, StatsTuple > ensemble_set1 = PotentialEnsemblesToMap(ensemble1);
	std::map<vector<int>, StatsTuple > ensemble_set2 = PotentialEnsemblesToMap(ensemble2);

	std::set<std::vector<int > > ensemble_intersection;

	std::map<vector<int>, StatsTuple >::iterator set_it1;
	std::map<vector<int>, StatsTuple >::iterator set_it2;

	string stat_func;
			double stat_result = 0.0;
			std::vector<long double > trade_pnl_vect;

	w_acc acc;
	w_acc acc1;

	//FOR THE CORRECT MEDIAN WHEN THERE IS AN EVEN NUMBER OF DATA POINTS
	//bacc::accumulator_set<double,
	//bacc::stats<bacc::tag::median(bacc::with_density) > >
	//    acc_median( bacc::density_cache_size = 4, bacc::density_num_bins = 4 );
	std::map<std::string, std::function<double()> > stats {
		{ "min",   [&acc] () { return bacc::min(acc);  }},
		{ "mean",  [&acc] () { return bacc::mean(acc); }},
		{ "median", [&acc] () { return bacc::median(acc); }},
		{ "max",   [&acc] () { return bacc::max(acc);  }},
		{ "range", [&acc] () { return (bacc::max(acc) - bacc::min(acc)); }},
		{ "var",   [&acc] () {
			int n = bacc::count(acc);
			double f = (static_cast<double>(n) / (n - 1));
			return f * bacc::variance(acc);
		}},
		{ "sd",    [&stats] () { return std::sqrt(stats["var"]()); }}
	};



	/*for(auto it = ensemble_set.begin(); it != ensemble_set.end(); ++it){
		set_it1 = ensemble_set1.find(it->first);
		if(set_it1 != ensemble_set1.end()){
			cout<<"AnalizeEnsembles ensemble "<<set_it1->first<<" "<<it->first<<endl;
			cout<<"Percent Correct "<<std::get<0>(set_it1->second)<<" Ensemble n_runs "<<std::get<1>(set_it1->second)<<" Acctual_n_runs "<<std::get<2>(set_it1->second)<<
									" best_loser_runs "<<std::get<3>(set_it1->second)<<" longest_total_losing_streak "<<std::get<4>(set_it1->second)<<
									" start_losing "<<std::get<5>(set_it1->second).date()<<" end_losing "<<std::get<6>(set_it1->second).date()<<" highest_wl "
									<<std::get<7>(set_it1->second)<<" wl "<<std::get<8>(set_it1->second)<<
									" "<<std::get<9>(set_it1->second)<<" Date Highest WL "<<std::get<10>(set_it1->second).date()<<endl;
			cout<<"Percent Correct "<<std::get<0>(it->second)<<" Ensemble n_runs "<<std::get<1>(it->second)<<" Acctual_n_runs "<<std::get<2>(it->second)<<
												" best_loser_runs "<<std::get<3>(it->second)<<" longest_total_losing_streak "<<std::get<4>(it->second)<<
												" start_losing "<<std::get<5>(it->second).date()<<" end_losing "<<std::get<6>(it->second).date()<<" highest_wl "
												<<std::get<7>(it->second)<<" wl "<<std::get<8>(it->second)<<
												" "<<std::get<9>(it->second)<<" Date Highest WL "<<std::get<10>(it->second).date()<<endl;

		}
	}*/

	for(auto it = ensemble_set.begin(); it != ensemble_set.end(); ++it){
		set_it1 = ensemble_set1.find(it->first);
		set_it2 = ensemble_set2.find(it->first);
		//if(set_it1 != ensemble_set1.end())
		//	cout<<"AnalizeEnsembles ensemble set_it1 "<<set_it1->first<<" "<<it->first<<endl;

		//if(set_it2 != ensemble_set2.end())
		//	cout<<"AnalizeEnsembles ensemble set_it2 "<<set_it2->first<<" "<<it->first<<endl;

		if(set_it1 != ensemble_set1.end() && set_it2 != ensemble_set2.end()){
			cout<<"AnalizeEnsembles ensemble "<<set_it1->first<<" "<<it->first<<" "<<set_it2->first<<endl;
			cout<<"Percent Correct "<<std::get<0>(it->second)<<" Ensemble n_runs "<<std::get<1>(it->second)<<" Acctual_n_runs "<<std::get<2>(it->second)<<
					" best_loser_runs "<<std::get<3>(it->second)<<" longest_total_losing_streak "<<std::get<4>(it->second)<<
					" start_losing "<<std::get<5>(it->second).date()<<" end_losing "<<std::get<6>(it->second).date()<<" highest_wl "
					<<std::get<7>(it->second)<<" wl "<<std::get<8>(it->second)<<
					" "<<std::get<9>(it->second)<<" Date Highest WL "<<std::get<10>(it->second).date()<<endl;

			cout<<"Percent Correct "<<std::get<0>(set_it1->second)<<" Ensemble n_runs "<<std::get<1>(set_it1->second)<<" Acctual_n_runs "<<std::get<2>(set_it1->second)<<
					" best_loser_runs "<<std::get<3>(set_it1->second)<<" longest_total_losing_streak "<<std::get<4>(set_it1->second)<<
					" start_losing "<<std::get<5>(set_it1->second).date()<<" end_losing "<<std::get<6>(set_it1->second).date()<<" highest_wl "
					<<std::get<7>(set_it1->second)<<" wl "<<std::get<8>(set_it1->second)<<
					" "<<std::get<9>(set_it1->second)<<" Date Highest WL "<<std::get<10>(set_it1->second).date()<<endl;
			cout<<"Percent Correct "<<std::get<0>(set_it2->second)<<" Ensemble n_runs "<<std::get<1>(set_it2->second)<<" Acctual_n_runs "<<std::get<2>(set_it2->second)<<
					" best_loser_runs "<<std::get<3>(set_it2->second)<<" longest_total_losing_streak "<<std::get<4>(set_it2->second)<<
					" start_losing "<<std::get<5>(set_it2->second).date()<<" end_losing "<<std::get<6>(set_it2->second).date()<<" highest_wl "
					<<std::get<7>(set_it2->second)<<" wl "<<std::get<8>(set_it2->second)<<
					" "<<std::get<9>(set_it2->second)<<" Date Highest WL "<<std::get<10>(set_it2->second).date()<<endl;

			//for_each(std::get<12>(set_it2->second).begin(),std::get<12>(set_it2->second).end(),[&] (long double i) {acc(i); cout<<"i"<<i<<endl;});

			for(auto it = std::get<11>(set_it1->second).begin(); it != std::get<11>(set_it1->second).end(); ++it){
				//cout<<"it"<<*it<<endl;
				acc(static_cast<double>(*it));
			}

			/* Todo
			 * Split this off into its own function this way I can do past year and all years
			 */
			for (auto s : stats){
				stat_func = s.first;
				stat_result = s.second();
				cout<<" "<<stat_func<<" "<<stat_result;
			}
			cout<<" sharpe "<<stats["mean"]()/stats["sd"]();
			cout<<endl;
			acc = w_acc();
			//PlotAndSavePnL(set_it1->first,ensemble_set2);



		}
	}


	//cout<<"break "<<endl;

	//auto intersect = std::set_intersection(ensemble_set.begin(), ensemble_set.end(), ensemble_set1.begin(), ensemble_set1.end(), ensemble_intersection.begin());

}





#ifndef MAP

template<typename T, typename T1>
std::map<int, std::map<vector<int>, StatsTuple > > AnalizeEnsembles(vector<vector<int> > &combos, T &ensemble, T1 &labels, bool past_year ){


	//cout<<"AnalizeEnsembles"<<endl;


	typename std::remove_reference<decltype(ensemble)>::type data;
	typedef decltype(data) pred_type;
	typename pred_type::iterator cBegin= ensemble.begin();
	typename pred_type::iterator cEnd = ensemble.end();

	//ensemble//T::iterator cBegin = ensemble.begin();
	//T::iterator cEnd = ensemble.end();


	long ud = 0;
	long prev_ud = 0;
	int acctual_n_runs = 0;


	std::map<boost::posix_time::ptime, long > it;

	w_acc acc;

	//FOR THE CORRECT MEDIAN WHEN THERE IS AN EVEN NUMBER OF DATA POINTS
	//bacc::accumulator_set<double,
	//bacc::stats<bacc::tag::median(bacc::with_density) > >
	//    acc_median( bacc::density_cache_size = 4, bacc::density_num_bins = 4 );
	std::map<std::string, std::function<double()> > stats {
		{ "min",   [&acc] () { return bacc::min(acc);  }},
		{ "mean",  [&acc] () { return bacc::mean(acc); }},
		{ "median", [&acc] () { return bacc::median(acc); }},
		{ "max",   [&acc] () { return bacc::max(acc);  }},
		{ "range", [&acc] () { return (bacc::max(acc) - bacc::min(acc)); }},
		{ "var",   [&acc] () {
			int n = bacc::count(acc);
			double f = (static_cast<double>(n) / (n - 1));
			return f * bacc::variance(acc);
		}},
		{ "sd",    [&stats] () { return std::sqrt(stats["var"]()); }}
	};

	int winner = 0;
	int count = 0;
	float best = 0.0;
	std::vector<int> best_ensemble;
	std::vector<int> ensemble_best_runs;
	std::vector<int> ensemble_least_loser_runs;
	std::vector<int> ensemble_shortest_losing_streak;
	int best_runs = 0;
	int ensemble_n_runs = 0;

	float percent_correct = 0.000000000000;
	int best_loser_runs = labels.size();
	int loser_runs_count=0;
	int loser_runs = 0;

	int best_total_losing_streak = labels.size();
	int longest_total_losing_streak =0;
	int wl=0;//wl++ for winner wl -- for losers
	int highest_wl = 0;
	boost::posix_time::ptime highest_wl_date;
	boost::posix_time::ptime start_losing;
	boost::posix_time::ptime start_losing1;
	boost::posix_time::ptime end_losing;
	bool end_losing_streak = true;
	int highest_wl_when_losses_started = 0;

	std::vector<long double > total_pnl_vect;
	vector<long double > trade_pnl_vect;

	std::map<std::vector<int >, StatsTuple> potential_ensembles;
	std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles1;

	float sqrd_diff = 0.0000000000;
	float rmse = 0.0000000000;

	/*TODO:
	 * for std::vector<int> best_ensemble should be vector<std::vector<int>> to handle ties
	 * 	this needs to be done for everything that might be ties
	 * for losing runs need start date and end date
	 * need like a longest losing streak
	 * calculate PnL
	 */

	for(; cBegin != cEnd; ++cBegin){


		for(int it = 0; it < cBegin->second.size(); ++it){
			sqrd_diff += std::pow((cBegin->second[it] - labels[it]),2);

		}
		sqrd_diff = sqrd_diff/cBegin->second.size();
		rmse = std::sqrt(sqrd_diff);

	}



	cout<<"RMSE "<<rmse<<endl;

	//can use this to check sequence switching  std::unique (myvector.begin(), myvector.end());

	total_pnl_vect.clear();
	trade_pnl_vect.clear();

	/*cout<<"Best_ensemble "<<best_ensemble<<"_"<<best<<endl;
	//cout<<"Best Range Model "<<best_range_model<<" "<<best_range<<endl;
	cout<<"best_runs_Ensemble "<< ensemble_best_runs<<"_"<<best_runs<<endl;
	cout<<"Ensemble shortest losing streak "<<ensemble_least_loser_runs<<"_"<<best_loser_runs<<endl;
	cout<<"Ensemble shortest total losing streak "<<ensemble_shortest_losing_streak<<"_"<<best_total_losing_streak<<endl;*/


	return potential_ensembles1;

}
template std::map<int, std::map<vector<int>, StatsTuple > >  AnalizeEnsembles<std::map<std::vector<int >, std::vector<float> >, std::vector<float> >(vector<vector<int> > &, std::map<std::vector<int >, std::vector<float> > &, std::vector<float> &, bool);

/*
 * @brief: lines up data into take in a vector of multimap, vector<long double and returns a multimap, vector<long double> in
 * 		you have a bunch of different features that need to be redeuced down to one container
 * 		so if you have predictions from 10 different models that are in 10 different containers this will reduce them down to just one container
 * 		these features are then used in a model
 */
std::multimap<boost::posix_time::ptime, std::vector<long double>> BuildFnlData(std::vector<std::multimap<boost::posix_time::ptime, std::vector<long double> > > &data_to_combined){


	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cBegin = data_to_combined[0].begin();
	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cEnd = data_to_combined[0].end();

	std::multimap<boost::posix_time::ptime, std::vector<long double>> FnlData;
	boost::posix_time::ptime date = cBegin->first;
	std::vector<long double > tmpVect;

	for(int i =data_to_combined.size()-1; i >=0; i--){
		if(data_to_combined[i].begin()->first > date){
			cBegin = data_to_combined[i].begin();
			cEnd = data_to_combined[i].end();
			date = cBegin->first;
		}

	}

	//data_to_combined[0][cBegin->first]-
	for(; cBegin!=cEnd; cBegin++){
		date = cBegin->first;
		//cout<<"date "<<date<<endl;
		//for(auto it =cBegin->second.begin(); it!= cBegin->second.end(); it++  )
		//	{tmpVect.push_back(*it); cout<<"it "<<*it<<endl;}
		for(int i =data_to_combined.size()-1; i >=0; i--){
			//cout<<"i "<<i<<endl;
			//auto vIt = data_to_combined[i].find(date)->second.end();
			//vIt--;
			for(auto vIt = data_to_combined[i].find(date)->second.begin(); vIt!=data_to_combined[i].find(date)->second.end(); ++vIt)
			{tmpVect.push_back(*vIt); }
		}

		FnlData.insert(std::pair<boost::posix_time::ptime, std::vector<long double>>
				(date,tmpVect));
		tmpVect.clear();


	}
	//Print(FnlData);

	return FnlData;

}

/*
 * @brief: lines up data into take in a vector of multimap, vector<long double and returns a multimap, vector<long double> in
 * 		you have a bunch of different features that need to be redeuced down to one container
 * 		so if you have predictions from 10 different models that are in 10 different containers this will reduce them down to just one container
 * 		these features are then used in a model
 */
std::vector<std::vector<long double> > BuildFnlDataVect(std::vector<std::multimap<boost::posix_time::ptime, std::vector<long double> > > &data_to_combined){

	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cBegin = data_to_combined[0].begin();
	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cEnd = data_to_combined[0].end();

	std::vector<std::vector<long double> > FnlData;
	boost::posix_time::ptime date;
	std::vector<long double > tmpVect;

	for(; cBegin!=cEnd; cBegin++){
		date = cBegin->first;
		//for(auto it =cBegin->second.begin(); it!= cBegin->second.end(); it++  )
		//	{tmpVect.push_back(*it); cout<<"it "<<*it<<endl;}
		for(int i =data_to_combined.size()-1; i >=0; i--){
			//cout<<"i "<<i<<endl;
			//auto vIt = data_to_combined[i].find(date)->second.end();
			//vIt--;
			for(auto vIt = data_to_combined[i].find(date)->second.begin(); vIt!=data_to_combined[i].find(date)->second.end(); ++vIt)
			{tmpVect.push_back(*vIt); }
		}

		FnlData.push_back(tmpVect);
		tmpVect.clear();


	}
	//Print(FnlData);

	return FnlData;
}


/*
 * @brief: puts the predictions from multiple models into one vector were the index number corresponds to the model number
 * 			and the model number is just the order in which the config file was opened, i.e no special meaning to the model number
 * 			after this you would then call another function to iterate through the vector to either average or find the most
 * 			frequent to decide what the prediction of the ensemble would be
 *
 */
template<typename T>
void BuildAllPredVectFnl(T &ensemble){


	uint model_number = 0;
	uint model_least_data_points = ensemble.all_pred_vect_tmp[model_number].size();
	for(uint i = model_number; i<ensemble.all_pred_vect_tmp.size(); ++i){
		if(ensemble.all_pred_vect_tmp[i].size()<model_least_data_points){
			model_least_data_points = ensemble.all_pred_vect_tmp[i].size();
			model_number=i;
		}

	}

	//std::multimap<boost::posix_time::ptime,  float >::iterator cBegin = ensemble.all_pred_vect_tmp[model_number].begin();
	//std::multimap<boost::posix_time::ptime, float >::iterator cEnd = ensemble.all_pred_vect_tmp[model_number].end();

	int cBegin = model_number;
	int cEnd = ensemble.all_pred_vect_tmp[model_number].size();

	//std::multimap<boost::posix_time::ptime, std::vector<long double>> FnlData;
	//boost::posix_time::ptime date = cBegin->first;
	std::vector<float > tmpVect(ensemble.all_pred_vect_tmp.size());
	//long pred_value =0;

	ensemble.all_pred_fnl.resize(cEnd,std::vector<std::vector<float > > (ensemble.all_pred_vect_tmp.size()));

	//here, trying to get this to work, this issue below is that I pred_type should be a vector<vector<float > >
	//typename std::remove_reference<decltype(ensemble.all_pred_vect_tmp[0])>::type pred_type;
	typename  std::remove_reference<decltype(ensemble.all_pred_vect_tmp[0])>::type pred_value;

	//cout<<"start date "<<date<<endl;
	for(uint i = 0; i<ensemble.all_pred_vect_tmp.size(); ++i){
		for(cBegin = model_number; cBegin!=cEnd; ++cBegin){
			//pred_value.push_back(ensemble.all_pred_vect_tmp[cBegin][i]);//pred_value = ensemble.all_pred_vect_tmp[i];
			ensemble.all_pred_fnl[cBegin][i] = ensemble.all_pred_vect_tmp[i][cBegin];//.push_back(pred_value);
			//tmpVect.push_back(pred_value);
			//how to build ensembles here????
		}

		//caffe2::all_pred_vect_fnl.insert(caffe2::all_pred_vect_fnl.begin(), tmpVect));

		//for(a in Combinations}{
		//for(b in a){
		//	if(tmpVect[b] == 0)// can't sum and divide becuae if I have a 2 it won't sum evenly ensemble_pred += tmpVect[b]
		//		zero++;
		//	else
		//		one++;
		//predvect.push_back
		//ensmeble[a].pus_pushback{ ,

	//	ensemble.all_pred_fnl.push_back(pred_value);
		pred_value.clear();
		//ensemble.all_pred_fnl[cBegin].push_back(tmpVect);
		//tmpVect.clear();
	}

	//std::multimap<boost::posix_time::ptime, std::vector< float > >::iterator end_date = ensemble.all_pred_fnl.end();
	//end_date--;
	//cout<<"Ensemble Start Dates "<<ensemble.all_pred_fnl.begin()->first<<" End Date "<<end_date->first<<endl;
	//cout<<"break "<<endl;

}
template void BuildAllPredVectFnl<Ensemble<std::vector<std::vector<float > >, std::vector<std::vector<float > >> >(Ensemble<std::vector<std::vector<float > >, std::vector<std::vector<float > >> &);


/*
 * Make final prediction
 *
 */
template<typename T>
void PredictEnsemble(T &all_pred_fnl){




	std::multimap<boost::posix_time::ptime, std::vector<long>>::iterator cBegin = all_pred_fnl.begin();
	std::multimap<boost::posix_time::ptime, std::vector<long>>::iterator cEnd = all_pred_fnl.end();

	cout<<all_pred_fnl.begin()->second.size()<<endl;
	std::vector<int > n_models(all_pred_fnl.begin()->second.size());
	//std::generate (n_models.begin(), n_models.end(), UniqueNumber);
	for(uint i =0; i<n_models.size();++i)
		n_models[i] =i;

	vector<vector<int> > combos;// = Combinations(n_models,7);
	std::vector<int> my_ensemble = {0,2,7,10,17,21,25};//was buy for
	//std::vector<int> my_ensemble = {0,3,8,12,16,18,26};
	//std::vector<int> my_ensemble = {4,6,8,11,16,18,21};
	//std::vector<int> my_ensemble = {0,1,12,14,17,20,22};
	combos.push_back(my_ensemble);


	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long> > ensemble;
	std::multimap<boost::posix_time::ptime, long> ensemble_tmp;
	int count0 = 0;
	int count1 = 0;


	//iterate date
	for(; cBegin!=cEnd; ++cBegin){
		//iterate ensemble
		for(uint i = 0; i< combos.size(); ++i){
			//cout<<"model combo "<<combos[i]<<endl;
			//iterate models in enesemble to find what they predicted, 0,0,1 means they predicted sell
			for(uint j = 0; j<combos[i].size(); ++j){
				//cout<<"model "<<combos[i][j]<<" prediction "<<cBegin->second[combos[i][j]]<<endl;
				if(cBegin->second[combos[i][j]]==0)
					count0++;
				else
					count1++;
				//can I build the ensemble here and look at the samples at the same time?????
			}
			if(count0>count1){
				//ensemble_tmp.insert(std::pair<boost::posix_time::ptime, long >
				//(cBegin->first,0));
				ensemble[combos[i]].insert(std::pair<boost::posix_time::ptime, long > (cBegin->first,0));
			}
			else{
				//ensemble_tmp.insert(std::pair<boost::posix_time::ptime, long >
				//(cBegin->first,1));
				ensemble[combos[i]].insert(std::pair<boost::posix_time::ptime, long >(cBegin->first,1));
			}
			//cout<<"ensemble "<<ensemble[ensemble.size()-1]<<endl;
			//cout<<"All models pred "<<cBegin->second<<endl;
			count0 = 0;
			count1 = 0;
		}

	}

	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long> >::iterator eBegin = ensemble.begin();
	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long> >::iterator eEnd = ensemble.end();

	std::multimap<boost::posix_time::ptime, std::vector<long double> > Sprd;

	std::map<boost::posix_time::ptime, long > labels;
//this gets the labels	this shoule be a function parameter SprdData(caffe2::data_path+caffe2::spread_filename, Sprd,false);

	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator label_it;
	std::multimap<boost::posix_time::ptime, std::vector<long double >>::iterator Sprd_End = Sprd.end();

	long ud = 0;

	//put labels into a map
	for(auto date_it = eBegin->second.begin(); date_it !=eBegin->second.end(); ++date_it){
		label_it = Sprd.find(date_it->first);
		//cout<<"cBegin->first "<<cBegin->first<<endl;
		if(label_it != Sprd_End){
			ud = label_it->second[0];
			labels[date_it->first]=ud;
			cout<<"labels Date "<<labels.find(date_it->first)->first<<" ud "<<labels.find(date_it->first)->second<<
					" Sprd Pric "<<Sprd.find(date_it->first)->second<<endl;
		}
		else
			cout<<"Problem in PredictEnsemble Can't find date "<<label_it->first<<endl;
	}



	boost::posix_time::ptime date_it;

	cout<<ensemble[my_ensemble]<<endl;
	std::multimap<boost::posix_time::ptime, long>::iterator ensemble_it = ensemble[my_ensemble].begin();
	for(; ensemble_it !=ensemble[my_ensemble].end(); ++ensemble_it){
		if(ensemble_it != ensemble[my_ensemble].begin()){
			cout<<"Date "<<ensemble_it->first<<" Pred "<<ensemble_it->second<<" labels Date "<<labels.find(date_it)->first<<" ud "<<labels.find(date_it)->second<<
				" Sprd Pric "<<Sprd.find(date_it)->second<<endl;
		}
		date_it = ensemble_it->first;
	}
	cout<<" labels Date "<<labels.find(date_it)->first<<" ud "<<labels.find(date_it)->second<<
					" Sprd Pric "<<Sprd.find(date_it)->second<<endl;


	ensemble_it = ensemble[my_ensemble].end();
	ensemble_it--;
	if(ensemble_it->second == 1)
		cout<<"Todays Prediction is, Buy "<<endl;
	else if(ensemble_it->second == 0)
		cout<<"Todays Prediction is, Sell "<<endl;
	else
		cout<<"Something wrong with Prediction PredictEnsemble "<<endl;





}

/*
 * @brief: used for going through combos one at a time, more used for individual models
 */
template<typename T>
void BuildEnsemble(T &all_pred_fnl){



	//PredictEnsemble();

	std::multimap<boost::posix_time::ptime, std::vector<long>>::iterator cBegin = all_pred_fnl.begin();
	std::multimap<boost::posix_time::ptime, std::vector<long>>::iterator cEnd = all_pred_fnl.end();

	cout<<all_pred_fnl.begin()->second.size()<<endl;
	std::vector<int > n_models(all_pred_fnl.begin()->second.size());
	//std::generate (n_models.begin(), n_models.end(), UniqueNumber);
	for(uint i =0; i<n_models.size();++i)
		n_models[i] =i;

	vector<vector<int> > combos = Combinations(n_models,7);
	std::vector<int> my_ensemble = {0,1,2,3,4,13,14,16,18,19,20};//was buy for
	std::vector<int> my_ensemble1 = {0,1,2,3,4,13,14,16,18,19,21};
	std::vector<int> my_ensemble2 = {0,1,2,3,4,13,14,16,18,19,22};
	std::vector<int> my_ensemble3 = {0,1,2,3,4,13,14,16,18,19,23};
	std::vector<int> my_ensemble4 = {0,1,2,3,4,13,14,16,18,19,24};
	combos.push_back(my_ensemble);
	combos.push_back(my_ensemble1);
	combos.push_back(my_ensemble2);
	combos.push_back(my_ensemble3);
	combos.push_back(my_ensemble4);


	//std::vector<int> my_ensemble = {6,7,11,12,14,21,25};//was buy for
	//std::vector<int> my_ensemble = {0,6,11,17,22,23,26};
	//std::vector<int> my_ensemble = {0,4,12,13,17,18,20};
	//std::vector<int> my_ensemble = {0,1,5,14,17,18,23};
	//std::vector<int> my_ensemble = {4,5,13,14,17,18,20};
	//combos.push_back(my_ensemble);

	//std::vector<int> my_ensemble = {6,17,20,21,25};//was buy for
	//std::vector<int> my_ensemble = {0,2,14,21,25};
	//std::vector<int> my_ensemble = {5,6,12,13,25};
	//std::vector<int> my_ensemble = {0,5,6,15,19};
	//std::vector<int> my_ensemble = {0,5,13,14,18};
	//combos.push_back(my_ensemble);


	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long > > ensemble;
	std::multimap<boost::posix_time::ptime, long> ensemble_tmp;
	int count0 = 0;
	int count1 = 0;


//
	//iterate date
	for(; cBegin!=cEnd; ++cBegin){
		//iterate ensemble
		for(uint i = 0; i< combos.size(); ++i){
			//cout<<"model combo "<<combos[i]<<endl;
			//iterate models in enesemble to find what they predicted, 0,0,1 means they predicted sell
			for(uint j = 0; j<combos[i].size(); ++j){
				//cout<<"model "<<combos[i][j]<<" prediction "<<cBegin->second[combos[i][j]]<<endl;
				if(cBegin->second[combos[i][j]]==0)
					count0++;
				else
					count1++;
				//can I build the ensemble here and look at the samples at the same time?????
			}
			if(count0>count1){
				//ensemble_tmp.insert(std::pair<boost::posix_time::ptime, long >
				//(cBegin->first,0));
				ensemble[combos[i]].insert(std::pair<boost::posix_time::ptime, long > (cBegin->first,0));
			}
			else{
				//ensemble_tmp.insert(std::pair<boost::posix_time::ptime, long >
				//(cBegin->first,1));
				ensemble[combos[i]].insert(std::pair<boost::posix_time::ptime, long >(cBegin->first,1));
			}
			//cout<<"ensemble "<<ensemble[ensemble.size()-1]<<endl;
			//cout<<"All models pred "<<cBegin->second<<endl;
			count0 = 0;
			count1 = 0;
		}

	}


	//iterate date
	//for(; cBegin!=cEnd; ++cBegin){
	//iterate ensemble
	for(uint i = 0; i< combos.size(); ++i){
		//cout<<"model combo "<<combos[i]<<endl;
		//iterate models in enesemble to find what they predicted, 0,0,1 means they predicted sell
		for(cBegin = all_pred_fnl.begin(); cBegin!=cEnd; ++cBegin){
			for(uint j = 0; j<combos[i].size(); ++j){
				//cout<<"model "<<combos[i][j]<<" prediction "<<cBegin->second[combos[i][j]]<<endl;
				if(cBegin->second[combos[i][j]]==0)
					count0++;
				else
					count1++;
				//can I build the ensemble here and look at the samples at the same time?????
			}
			if(count0>count1){
				//ensemble_tmp.insert(std::pair<boost::posix_time::ptime, long >
				//(cBegin->first,0));
				ensemble[combos[i]].insert(std::pair<boost::posix_time::ptime, long > (cBegin->first,0));
			}
			else{
				//ensemble_tmp.insert(std::pair<boost::posix_time::ptime, long >
				//(cBegin->first,1));
				ensemble[combos[i]].insert(std::pair<boost::posix_time::ptime, long >(cBegin->first,1));
			}
			//cout<<"ensemble "<<ensemble[ensemble.size()-1]<<endl;
			//cout<<"All models pred "<<cBegin->second<<endl;
			count0 = 0;
			count1 = 0;
		}


		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_all_years;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_past_year;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_all_years;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_past_year;

		//potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, false);
		//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
		//potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,false);
		//potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,true);

		FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_PnL_past_year,potential_ensembles_PnL_all_years);
		//FindIntersectingEnsemblesMap(potential_ensembles_PnL_all_years,potential_ensembles_PnL_past_year,potential_ensembles_all_years);

		//FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_past_year,potential_ensembles_PnL_all_years);

		std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long > > ensemble;

	}

	std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_all_years;
	std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_past_year;
	std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_all_years;
	std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_past_year;

//	potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, false);
	//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
	//potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,false);
	//potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,true);

	FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_PnL_past_year,potential_ensembles_PnL_all_years);
	//FindIntersectingEnsemblesMap(potential_ensembles_PnL_all_years,potential_ensembles_PnL_past_year,potential_ensembles_all_years);

	//FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_past_year,potential_ensembles_PnL_all_years);





}

template <typename T, typename T1>
void BuildEnsembleThreading(T &all_pred_fnl, T1 &labels){



	//PredictEnsemble();



	//cout<<all_pred_fnl.begin()->second.size()<<endl;
	std::vector<int > n_models(all_pred_fnl[0].size());//(all_pred_fnl.begin()->second.size());
	//std::generate (n_models.begin(), n_models.end(), UniqueNumber);
	for(uint i =0; i<n_models.size();++i)
		n_models[i] =i;

	vector<vector<int> > combos;// = Combinations(n_models,11);
	std::vector<int> my_ensemble = {0,1,2};//was buy for
	//std::vector<int> my_ensemble1 = {0,1,2,3,4,13,14,16,18,19,21};
	//std::vector<int> my_ensemble2 = {0,1,2,3,4,13,14,16,18,19,22};
	//std::vector<int> my_ensemble3 = {0,1,2,3,4,13,14,16,18,19,23};
	//std::vector<int> my_ensemble4 = {0,1,2,3,4,13,14,16,18,19,24};
	//combos.push_back(my_ensemble);
	//combos.push_back(my_ensemble1);
	//combos.push_back(my_ensemble2);
	//combos.push_back(my_ensemble3);
	//combos.push_back(my_ensemble4);


	//std::vector<int> my_ensemble = {6,7,11,12,14,21,25};//was buy for
	//std::vector<int> my_ensemble = {0,6,11,17,22,23,26};
	//std::vector<int> my_ensemble = {0,4,12,13,17,18,20};
	//std::vector<int> my_ensemble = {0,1,5,14,17,18,23};
	//std::vector<int> my_ensemble = {4,5,13,14,17,18,20};
	//combos.push_back(my_ensemble);

	//std::vector<int> my_ensemble = {6,17,20,21,25};//was buy for
	//std::vector<int> my_ensemble = {0,2,14,21,25};
	//std::vector<int> my_ensemble = {5,6,12,13,25};
	//std::vector<int> my_ensemble = {0,5,6,15,19};
	//std::vector<int> my_ensemble = {0,5,13,14,18};
	combos.push_back(my_ensemble);




	int n_cores = std::thread::hardware_concurrency();

	int combos_size = combos.size();

		//cout<<"modulus "<<std::modulus<double>((double)vect_size/(double)n_cores)<<endl;
		cout<<"modulus "<<combos_size/n_cores<<endl;
		cout<<"modulus "<<combos_size%n_cores<<endl;

		//std::div_t map_it_per_core= std::div(distance,n_cores);//std::div(data.size(),n_cores);
		//map_it_per_core();
		//data.size()/n_cores;
		int iter_per_core = combos_size/n_cores;
		int iter_per_core_remainder = combos_size%n_cores;
		//int iter_per_core = (distance - end)/n_cores;

		std::vector<std::thread> threads;

		int what_to_name = 0;//vect_size;

		int start =0;
		int end = 0;

		//iter = cEnd;
		//cBegin = cEnd;

		for(int i = 0; i< n_cores; ++i){

			//std::advance(cBegin, -map_it_per_core.quot);

			end = start + iter_per_core;


			//if(i == 2)
			try{
				cout<<"start "<<start<<" end "<<end<<endl;
				threads.push_back(std::thread(GoThroughCombos<T, T1>,start, end, std::ref(combos),std::ref(all_pred_fnl),std::ref(labels)));
			}catch(const char *msg){
				cerr << msg << endl;
			}

			start = end+1;

			//if(i == 2)
			//	break;

		}

		for(uint i = 0; i<threads.size(); ++i)
			{
			threads[i].join();
			cout<<"i "<<i<<endl;
			}








}
template void BuildEnsembleThreading< std::vector<std::vector<std::vector<float > > >, std::vector<float > >(std::vector<std::vector<std::vector<float > > >  &, std::vector<float > &);



/*
 * @brief: used to find which combination of models makes the best ensemble iterates through a
 * 			vector<vector<int> > which is various combinations of models
 *
 */
template<typename T, typename T1>
void GoThroughCombos(int start, int end, vector<vector<int> > &combos, T &all_pred_fnl, T1 &labels){


	int cBegin = 0;//all_pred_fnl.begin();
	int cEnd = all_pred_fnl.size();




	//std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long > > ensemble;
	typename std::remove_reference<decltype(all_pred_fnl[0][0])>::type data(1);
	typedef decltype(data) pred_type;
	std::map<std::vector<int >, pred_type  > ensemble;

	std::multimap<boost::posix_time::ptime, long> ensemble_tmp;
	float count0 = 0;
	//int count1 = 0;



	//for when I want to run just 1 model that I have already picked, like just to make some charts for a model
	if(combos.size() == 1 && end == 0)
		end = 1;

	for(int i = start; i< end; ++i){
		//cout<<"model combo "<<combos[i]<<endl;
		//iterate models in enesemble to find what they predicted, 0,0,1 means they predicted sell
		for(cBegin = 0; cBegin!=cEnd; ++cBegin){
			for(uint j = 0; j<combos[i].size(); ++j){
				//cout<<"model "<<combos[i][j]<<" prediction "<<cBegin->second[combos[i][j]]<<endl;
				count0 += all_pred_fnl[cBegin][j][0];
			}

				ensemble[combos[i]].push_back(count0/(combos[i].size()));

			//cout<<"ensemble "<<ensemble[ensemble.size()-1]<<endl;
			//cout<<"All models pred "<<cBegin->second<<endl;
			count0 = 0;
			//count1 = 0;
		}


		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_all_years;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_past_year;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_all_years;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_past_year;

		try
		{
			potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, labels, false);
		//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
		//potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,false);
		//potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,true);

		if(potential_ensembles_all_years.size()>0&&  potential_ensembles_PnL_all_years.size()>0&&  potential_ensembles_PnL_past_year.size()>0){
			//cout<<"model combo "<<combos[i]<<" i "<<i<<endl;
			FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_PnL_past_year,potential_ensembles_PnL_all_years);
		}
		//FindIntersectingEnsemblesMap(potential_ensembles_PnL_all_years,potential_ensembles_PnL_past_year,potential_ensembles_all_years);

		//FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_past_year,potential_ensembles_PnL_all_years);

		//std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long > > ensemble;

		potential_ensembles_all_years.clear();
		//potential_ensembles_past_year.clear();
		potential_ensembles_PnL_all_years.clear();
		potential_ensembles_PnL_past_year.clear();
		}catch(const char *msg){
			cerr << msg << endl;
		}

		ensemble.clear();

	}

		//std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_all_years;
		//std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_past_year;
		//std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_all_years;
		//std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_past_year;

		//potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, labels, false);
		//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
		//potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,false);
		//potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,true);

//		FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_PnL_past_year,potential_ensembles_PnL_all_years);
		//FindIntersectingEnsemblesMap(potential_ensembles_PnL_all_years,potential_ensembles_PnL_past_year,potential_ensembles_all_years);

		//FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_past_year,potential_ensembles_PnL_all_years);

}
template void GoThroughCombos<std::vector<std::vector<std::vector<float > > >, std::vector<float >  >(int , int , vector<vector<int> > &, std::vector<std::vector<std::vector<float > > > &, std::vector<float > &);



template<typename T>
void GoThroughCombosBinary(int start, int end, vector<vector<int> > &combos, T &all_pred_fnl){


	std::multimap<boost::posix_time::ptime, std::vector<long>>::iterator cBegin = all_pred_fnl.begin();
	std::multimap<boost::posix_time::ptime, std::vector<long>>::iterator cEnd = all_pred_fnl.end();

	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long > > ensemble;
	std::multimap<boost::posix_time::ptime, long> ensemble_tmp;
	int count0 = 0;
	int count1 = 0;



	//for when I want to run just 1 model that I have already picked, like just to make some charts for a model
	if(combos.size() == 1 && end == 0)
		end = 1;

	for(int i = start; i< end; ++i){
		//cout<<"model combo "<<combos[i]<<endl;
		//iterate models in enesemble to find what they predicted, 0,0,1 means they predicted sell
		for(cBegin = all_pred_fnl.begin(); cBegin!=cEnd; ++cBegin){
			for(uint j = 0; j<combos[i].size(); ++j){
				//cout<<"model "<<combos[i][j]<<" prediction "<<cBegin->second[combos[i][j]]<<endl;
				if(cBegin->second[combos[i][j]]==0)
					count0++;
				else
					count1++;
				//can I build the ensemble here and look at the samples at the same time?????
			}
			if(count0>count1){
				//ensemble_tmp.insert(std::pair<boost::posix_time::ptime, long >
				//(cBegin->first,0));
				ensemble[combos[i]].insert(std::pair<boost::posix_time::ptime, long > (cBegin->first,0));
			}
			else{
				//ensemble_tmp.insert(std::pair<boost::posix_time::ptime, long >
				//(cBegin->first,1));
				ensemble[combos[i]].insert(std::pair<boost::posix_time::ptime, long >(cBegin->first,1));
			}
			//cout<<"ensemble "<<ensemble[ensemble.size()-1]<<endl;
			//cout<<"All models pred "<<cBegin->second<<endl;
			count0 = 0;
			count1 = 0;
		}


		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_all_years;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_past_year;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_all_years;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_past_year;

		try
		{//potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, false);
		//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
		//potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,false);
		//potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,true);

		if(potential_ensembles_all_years.size()>0&&  potential_ensembles_PnL_all_years.size()>0&&  potential_ensembles_PnL_past_year.size()>0){
			//cout<<"model combo "<<combos[i]<<" i "<<i<<endl;
			FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_PnL_past_year,potential_ensembles_PnL_all_years);
		}
		//FindIntersectingEnsemblesMap(potential_ensembles_PnL_all_years,potential_ensembles_PnL_past_year,potential_ensembles_all_years);

		//FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_past_year,potential_ensembles_PnL_all_years);

		//std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long > > ensemble;

		potential_ensembles_all_years.clear();
		//potential_ensembles_past_year.clear();
		potential_ensembles_PnL_all_years.clear();
		potential_ensembles_PnL_past_year.clear();
		}catch(const char *msg){
			cerr << msg << endl;
		}

		ensemble.clear();

	}

		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_all_years;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_past_year;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_all_years;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_past_year;

//		potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, false);
		//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
		//potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,false);
		//potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,true);

		FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_PnL_past_year,potential_ensembles_PnL_all_years);
		//FindIntersectingEnsemblesMap(potential_ensembles_PnL_all_years,potential_ensembles_PnL_past_year,potential_ensembles_all_years);

		//FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_past_year,potential_ensembles_PnL_all_years);

}

namespace CPCO{
template <typename T>
//this is more of ReadConfig and build data
void ReadConfig(string path_file_name, T ens){

	cout<<"CPCO"<<endl;
	int count = 0;
	//std::vector<std::string> filename;
	//filename=iterdir(file_path);

	//Search for the file of given name
	//string file_name=fileName;//"OFER_VDA_BMF_20170814.TXT";
	//const char *name = file_name.c_str();
	/*{
		std::ifstream  data(path_file_name);

		std::string line;

		while(std::getline(data,line))
		{
			if(count ==12)
				ens.model_name = line;
			if(count  == 1)
				ens.no_train_thresh = std::stof(line);
			if(count  == 2)
				ens.all_thresh = std::stof(line);
			else if(count ==3)
				ens.feature_vars = line;
			else if(count ==4)
				ens.TRTEVA = std::stoi(line);
			else if(count == 5)
				ens.vc = std::stoi(line);
			else if(count ==6)
				ens.n_hide[0] = std::stoi(line);
			else if(count ==7)
				ens.n_hide1 = std::stoi(line);
			else if(count ==8)
				ens.n_hide2 = std::stoi(line);
			else if(count ==9)
				ens.n_hide3 = std::stoi(line);
			else if(count == 10)
				ens.model_path = line;
			else if(count == 11)
				ens.data_path = line;

			count++;

			//cout<<"ReadConfig "<<line<<endl;

			//this->allData.append(line);
			std::stringstream  lineStream(line);
			std::string        cell;
			while(std::getline(lineStream,cell,','))
			{
				// You have a cell!!!!
			}
		}
	}

	bool ud = false;


	ens.data_path = "/home/ryan/Energy/CPCOModelsARConfig/";
	ens.model_path = "/home/ryan/Energy/CPCOModelsARCaffe2/";

	ens.chart_path = ens.data_path + "PnLCharts/";
	ens.init_model_name = "init"+ens.model_name;
	//cout<<"ens.model_name "<<ens.model_name<<endl;
	ens.n_classes =2;
	ens.learning_rate =-.00003;




	if(ens.feature_vars.find("UD")!=string::npos)
		ud = true;

	ens.spread_filename = "CPCODaily.csv";
	std::multimap<boost::posix_time::ptime, std::vector<long double>> Sprd;
	MyData::SprdData(ens.data_path+"CPCODaily.csv", Sprd, ud);
	std::multimap<boost::posix_time::ptime, std::vector<long double>> A;
	MyData::InputVar(ens.data_path+"CPCO_CPDaily.csv", A);
	std::multimap<boost::posix_time::ptime, std::vector<long double>> B;
	MyData::InputVar(ens.data_path+"CPCO_CODaily.csv", B);
	std::multimap<boost::posix_time::ptime, std::vector<long double>> C;
	MyData::InputVar(ens.data_path+"CPCODaily.csv", C);
	std::multimap<boost::posix_time::ptime, std::vector<long double>> D;
	MyData::InputVar(ens.data_path+"CPCO_CPDaily.csv", D);
	std::multimap<boost::posix_time::ptime, std::vector<long double>> E;
	MyData::InputVar(ens.data_path+"CPCO_CODaily.csv", E);

	if(ens.vc ==1){

		if(ud == false)
			MyData::EmbedNoReturn(Sprd,3,55);
		MyData::EmbedNoReturn(A,3,50);
		MyData::EmbedNoReturn(B,3,55);
		MyData::EmbedNoReturn(C,55,3,true);
		MyData::EmbedNoReturn(D,50,3,true);
		MyData::EmbedNoReturn(E,55,3,true);
	}
	else if(ens.vc ==2){
		if(ud == false)
			MyData::EmbedNoReturn(Sprd,3,27);
		MyData::EmbedNoReturn(A,3,25);
		MyData::EmbedNoReturn(B,3,27);
		MyData::EmbedNoReturn(C,27,3,true);
		MyData::EmbedNoReturn(D,25,3,true);
		MyData::EmbedNoReturn(E,27,3,true);
	}
	else if(ens.vc == 3){
		if(ud == false)
			MyData::EmbedNoReturn(Sprd,4,14);
		MyData::EmbedNoReturn(A,4,12);
		MyData::EmbedNoReturn(B,4,14);
		MyData::EmbedNoReturn(C,14,4,true);
		MyData::EmbedNoReturn(D,12,4,true);
		MyData::EmbedNoReturn(E,14,4,true);
	}



	std::vector<std::multimap<boost::posix_time::ptime, std::vector<long double> > > data;

	//used to find if we have a D feature
	size_t n_D = std::count(ens.feature_vars.begin(), ens.feature_vars.end(), 'D');

	data.push_back(Sprd);
	if(ens.feature_vars.find("A")!=string::npos)
		data.push_back(A);
	if(ens.feature_vars.find("B")!=string::npos)
		data.push_back(B);
	if(ens.feature_vars.find("C")!=string::npos)
		data.push_back(C);
	if((ens.feature_vars.find("UD")!=string::npos && n_D==2) || (ens.feature_vars.find("UD")==string::npos && n_D == 1))//feature_vars.find("D"))
		data.push_back(D);
	if(ens.feature_vars.find("E")!=string::npos)
		data.push_back(E);*/


	//auto FnlData = BuildFnlData(data);


	//MyData::SplitData<std::vector<std::vector<float> >,std::vector<std::vector<float> > > fu;


	//fu.FnlDataToStruct(,FnlData,fu.split_or_not::no_split)


	//Results.allBuys = std::accumulate(AllData.Labels.begin(), AllData.Labels.end(), 0);
	//Results.allBuys = Results.allBuys/(AllData.Labels.size()-1);

	//ens.n_features = FnlData.begin()->second.size()-1;

	//cout<<"Model "<<ens.model_name<<endl;
	//TestModel(ens);
	AllData = Data();

}

void ModelNames(){

	//I think just use config file becuase I need to parse the featur vars
	// and VC

	std::vector<std::string> model_names;

	//have to open data and build model and put into prediction vect
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TETRVA/CPCOC2_BA2_TETRVA_VC2_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TETRVA/CPCOC2_DCAa3_TETRVA_VC2_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TETRVA/CPCOC2_DCAUDa3_TETRVA_VC1_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TETRVA/CPCOC2_ECB3_TETRVA_VC1_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TETRVA/CPCOC2_EDBA3_TETRVA_VC3_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TETRVA/CPCOC2_EDC3_TETRVA_VC1_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TETRVA/CPCOC2_EDCBAb4_TETRVA_VC2_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TETRVA/CPCOC2_EDCBAUD2_TETRVA_VC3_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TETRVA/CPCOC2_EDCUD4_TETRVA_VC3_Config");

	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TRTEVA/CPCOC2_BA2_TRTEVA_VC2_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TRTEVA/CPCOC2_DCAa3_TRTEVA_VC2_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TRTEVA/CPCOC2_DCAUDa3_TRTEVA_VC1_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TRTEVA/CPCOC2_ECB3_TRTEVA_VC1_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TRTEVA/CPCOC2_EDBA3_TRTEVA_VC3_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TRTEVA/CPCOC2_EDC3_TRTEVA_VC1_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TRTEVA/CPCOC2_EDCBAb4_TRTEVA_VC2_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TRTEVA/CPCOC2_EDCBAUD2_TRTEVA_VC3_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/TRTEVA/CPCOC2_EDCUD4_TRTEVA_VC3_Config");

	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/VATRTE/CPCOC2_BA2_VATRTE_VC2_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/VATRTE/CPCOC2_DCAa3_VATRTE_VC2_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/VATRTE/CPCOC2_DCAUDa3_VATRTE_VC1_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/VATRTE/CPCOC2_ECB3_VATRTE_VC1_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/VATRTE/CPCOC2_EDBA3_VATRTE_VC3_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/VATRTE/CPCOC2_EDC3_VATRTE_VC1_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/VATRTE/CPCOC2_EDCBAb4_VATRTE_VC2_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/VATRTE/CPCOC2_EDCBAUD2_VATRTE_VC3_Config");
	model_names.push_back("/home/ryan/Energy/CPCOModelsARConfig/VATRTE/CPCOC2_EDCUD4_VATRTE_VC3_Config");


	//Ensemble<int, int> ens;
	Ensemble<std::vector<std::vector<float > >,std::vector<std::vector<float > > > ens;
	for(auto iter : model_names){
		ReadConfig(iter, ens);
		TestModel(ens);

	}


	//will hhave to make sure I clear vectors appropriatels
	//will need to add Spolit data and I don't know if ens is correct for the below
	BuildAllPredVectFnl(ens);

}

}

/*
 * @brief: runs a model and put the predictions from the model into the pred_vect of Ensemble
 *
 */
template<typename T,typename T1,typename T2>
void PredictAll(T &ens, T1 &features, T2 &labels ){


	caffe2::NetDef init_model, predict_model;

	string init_model_name = ens.model_path+ens.init_model_name();//+".pbtxt";
	string model_name = ens.model_path+ens.model_name;//+".pbtxt";

	//cout<<"model_path+init_model_name"<<model_path+init_model_name<<endl;
	CAFFE_ENFORCE(ReadProtoFromBinaryFile(init_model_name, &init_model));//
	CAFFE_ENFORCE(ReadProtoFromBinaryFile(model_name, &predict_model));//

	caffe2::Workspace workspace("tmp");



	//std::multimap<boost::posix_time::ptime, long > data;
	typename std::remove_reference<decltype(ens.all_pred_vect_tmp[0])>::type data;

	typedef decltype(ens.all_pred_vect_tmp[0][0]) pred_type;

	typename std::remove_reference<decltype(data[0])>::type pred_value;

	//typedef decltype(pred_value[0]) pred_type;

	//CAFFE_ENFORCE(workspace.RunNet(initModel.n));
	int nTrainBatches = features.size();//233;
	int minibatch_index = 0;

	{
		std::vector<int> dim({nTrainBatches,features[minibatch_index].size()});
		caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, features, minibatch_index,false);
	}

	{
		std::vector<int> dim({nTrainBatches,1});
		caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, labels,minibatch_index, false);
		//std::cout<<"Validate Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
	}

	caffe2::ModelUtil model(init_model, predict_model, model_name);
	CAFFE_ENFORCE(workspace.CreateNet(model.init.net));
	CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));


	//cout<<"model.init.net.name() "<<model.init.net.name()<<" model.predict.net.name() "<<model.predict.net.name()<<endl;
	CAFFE_ENFORCE(workspace.RunNet(model.init.net.name()));
	CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));

	//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;

	//Results.allResults = caffe2::BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];

	boost::posix_time::ptime date;
	for(int a = 0; a<nTrainBatches; a++){
		//cout<<"Date "<<AllData.DateToIter[a]<<" Predicted Value "<< BlobUtil(*workspace.GetBlob("argmax")).Get().data<long>()[a]<<endl;
		pred_value.push_back(caffe2::BlobUtil(*workspace.GetBlob("last_layer")).Get().data<float>()[a]);
		data.push_back(pred_value);
		pred_value.clear();
	}

	ens.all_pred_vect_tmp.push_back(data);

	if(Results.allResults>Results.best)
		Results.best =Results.allResults;
	//cout<<"All accuracy "<<Results.allResults<<" best "<<Results.best<<endl;


}
template void PredictAll<Ensemble<std::vector<std::vector<float > >, std::vector<std::vector<float > >>,std::vector<std::vector<float > >,std::vector<float > >(Ensemble<std::vector<std::vector<float > >, std::vector<std::vector<float > >> &,std::vector<std::vector<float > > &,std::vector<float > &);



template<typename T>
void PredictDay(caffe2::NetDef &init_model, caffe2::NetDef &predict_model, T &pred_vect, string model_name){

	caffe2::Workspace workspace("tmp");

	std::multimap<boost::posix_time::ptime, long > data;
	//1. is used to save the predicted values
	//1.std::multimap<boost::posix_time::ptime, std::vector<long double> > data;
	long pred_value = 0;
	//CAFFE_ENFORCE(workspace.RunNet(initModel.n));
	int nTrainBatches = AllData.Features.size();//233;
	int minibatch_index = 0;

	{
		std::vector<int> dim({nTrainBatches,AllData.Features[minibatch_index].size()});
		caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, AllData.Features, minibatch_index,false);
	}

	{
		std::vector<int> dim({nTrainBatches});
		caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, AllData.Labels,minibatch_index, false);
		//std::cout<<"Validate Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
	}

	caffe2::ModelUtil model(init_model, predict_model, model_name);
	CAFFE_ENFORCE(workspace.CreateNet(model.init.net));
	CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));


	CAFFE_ENFORCE(workspace.RunNet(model.init.net.name()));
	CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));

	//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;

	Results.allResults = caffe2::BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];

	for(int a = 0; a<nTrainBatches; a++){
		//cout<<"Date "<<AllData.DateToIter[a]<<" Predicted Value "<< BlobUtil(*workspace.GetBlob("argmax")).Get().data<long>()[a]<<endl;
		pred_value = caffe2::BlobUtil(*workspace.GetBlob("argmax")).Get().data<long>()[a];
		data.insert(std::pair<boost::posix_time::ptime, long > (AllData.DateToIter[a],pred_value));

	}

	pred_vect.push_back(data);

	if(Results.allResults>Results.best)
		Results.best =Results.allResults;
	cout<<"All accuracy "<<Results.allResults<<" best "<<Results.best<<endl;


}

void PredictNonTrained(caffe2::NetDef &init_model, caffe2::NetDef &predict_model, string model_name){

	caffe2::Workspace workspace("tmp");

	//Validate Data
	int nTrainBatches = ValidateData.Features.size()-1;
	int minibatch_index = 0;


	{
		std::vector<int> dim({nTrainBatches,ValidateData.Features[minibatch_index].size()});
		caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim,ValidateData.Features, minibatch_index,false);
	}

	{
		std::vector<int> dim({nTrainBatches});
		caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, ValidateData.Labels,minibatch_index, false);
		//std::cout<<"Validate Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
	}

	caffe2::ModelUtil model(init_model, predict_model, model_name);
	CAFFE_ENFORCE(workspace.CreateNet(model.init.net));
	CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));


	CAFFE_ENFORCE(workspace.RunNet(model.init.net.name()));
	CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));



	Results.validateResults = caffe2::BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];


	//Test Data
	{
		std::vector<int> dim({nTrainBatches,TestData.Features[minibatch_index].size()});
		caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim,TestData.Features, minibatch_index,false);
	}

	{
		std::vector<int> dim({nTrainBatches});
		caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, TestData.Labels,minibatch_index, false);
		//std::cout<<"Validate Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
	}

	CAFFE_ENFORCE(workspace.RunNet(model.init.net.name()));
	CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));

	Results.testResults = caffe2::BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];

	Results.diffResults = std::abs(Results.validateResults-Results.testResults);
	Results.combinedResults = (Results.validateResults+Results.testResults)/2;

	if(Results.combinedResults>Results.bestNonTrained)
		Results.bestNonTrained = Results.combinedResults;

	cout<<"AllBuys "<<Results.allBuys<<endl;
	cout<<"ModelNoTrain "<<Results.combinedResults<<endl;
	//cout<<"Diff Reults nonTrained "<< Results.diffResults<<endl;

	if(Results.allBuys>.5){
		if(Results.combinedResults>Results.allBuys)
			Results.betterThan=true;
	}
	else{
		if(Results.combinedResults>1-Results.allBuys)
			Results.betterThan=true;
	}

	//(Results.betterThan == true) ? cout<<"BetterThan True"<<endl : cout<<"BetterThan False"<<endl;
	if(Results.combinedResults >=.57 && Results.diffResults>=.02){
		cout<<"Validate accuracy "<<Results.validateResults<<" Test Results "<<Results.testResults<<endl;
	}
}



template<typename T>
void TestModel(T &ens){

	caffe2::NetDef init_model, predict_model;

	string init_model_name = ens.model_path+ens.init_model_name()+".pbtxt";
	string model_name = ens.model_path+ens.model_name+".pbtxt";

	//cout<<"model_path+init_model_name"<<model_path+init_model_name<<endl;
	CAFFE_ENFORCE(ReadProtoFromTextFile(init_model_name, &init_model));
	CAFFE_ENFORCE(ReadProtoFromTextFile(model_name, &predict_model));



//	PredictAll(init_model, predict_model,ens.model_name,ens.all_pred_vect_tmp);
	//PredictDay(init_model, predict_model,all_pred_vect_tmp);



	//PredictNonTrained(init_model, predict_model);

}
template void TestModel<Ensemble<std::vector<std::vector<float > >, std::vector<std::vector<float > >>>(Ensemble<std::vector<std::vector<float > >, std::vector<std::vector<float > >> &);


#endif


