/*
 * EnsembleMap.cpp
 *
 *  Created on: Aug 20, 2019
 *      Author: ryan
 */

#include <Ensemble.h>







#ifdef MAP
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

	//what[0][cBegin->first]-
	for(; cBegin!=cEnd; cBegin++){
		date = cBegin->first;
		//cout<<"date "<<date<<endl;
		//for(auto it =cBegin->second.begin(); it!= cBegin->second.end(); it++  )
		//	{tmpVect.push_back(*it); cout<<"it "<<*it<<endl;}
		for(int i =data_to_combined.size()-1; i >=0; i--){
			//cout<<"i "<<i<<endl;
			//auto vIt = what[i].find(date)->second.end();
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
			//auto vIt = what[i].find(date)->second.end();
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

	std::multimap<boost::posix_time::ptime, long >::iterator cBegin = ensemble.all_pred_vect_tmp[model_number].begin();
	std::multimap<boost::posix_time::ptime, long >::iterator cEnd = ensemble.all_pred_vect_tmp[model_number].end();


	//std::multimap<boost::posix_time::ptime, std::vector<long double>> FnlData;
	boost::posix_time::ptime date = cBegin->first;
	std::vector<long > tmpVect;
	long pred_value =0;

	cout<<"start date "<<date<<endl;
	for(; cBegin!=cEnd; ++cBegin){
		for(uint i = 0; i<ensemble.all_pred_vect_tmp.size(); ++i){
			pred_value = ensemble.all_pred_vect_tmp[i].find(cBegin->first)->second;
			tmpVect.push_back(pred_value);
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

		ensemble.all_pred_fnl.insert(std::pair<boost::posix_time::ptime, std::vector<long>>
				(cBegin->first,tmpVect));
		tmpVect.clear();
	}

	std::multimap<boost::posix_time::ptime, std::vector< long > >::iterator end_date = ensemble.all_pred_fnl.end();
	end_date--;
	cout<<"Ensemble Start Dates "<<ensemble.all_pred_fnl.begin()->first<<" End Date "<<end_date->first<<endl;
	//cout<<"break "<<endl;

}


/*
 * @return std::map<int, std::map<vector<int>, StatsTuple > > were int is the key stat I am looking for vector<int> is the model numbers in the ensemble and StatsTuple is the the Tuple of statistics
 */
std::map<int, std::map<vector<int>, StatsTuple > > AnalizeEnsemblesPnL(vector<vector<int> > &combos, std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long> > &ensemble, string spread_filename, bool past_year){


	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long> >::iterator cBegin = ensemble.begin();
	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long> >::iterator cEnd = ensemble.end();


	w_acc acc;
	w_acc acc_winner;
	w_acc acc_loser;

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



	std::multimap<boost::posix_time::ptime, std::vector<long double> > Sprd;

	std::map<boost::posix_time::ptime, long > labels;
	try{
		MyData::SprdData(spread_filename, Sprd,false,true);
	}catch(std::exception &e){
		cout<<"didn't work AnalizeEnsemblesPnL "<<endl;
	}
	//SprdData(caffe2::data_path+caffe2::spread_filename, Sprd);

	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator label_it;
	std::multimap<boost::posix_time::ptime, std::vector<long double >>::iterator Sprd_End = Sprd.end();

	long ud = 0;
	long prev_ud = 0;
	int acctual_n_runs = 0;

	/*//put labels into a map and count runs
	//and performs a check
	for(auto date_it = cBegin->second.begin(); date_it !=cBegin->second.end(); ++date_it){
		label_it = Sprd.find(date_it->first);
		//cout<<"cBegin->first "<<cBegin->first<<endl;
		if(label_it != Sprd_End){
			ud = label_it->second[0];
			labels[date_it->first]=ud;
			if(prev_ud != ud && date_it != cBegin->second.begin())
				acctual_n_runs++;
			prev_ud = ud;
			//cout<<"labels Date"<<labels.find(date_it->first)->first<<" ud "<<labels.find(date_it->first)->second<<endl;
		}
		else
			cout<<"Problem in AnalizeEnsembles Can't find date "<<label_it->first<<endl;
	}

	std::map<boost::posix_time::ptime, long > it;*/


	//std::unique_copy(labels.begin(), labels.end(), unique_labels.begin());


	int winner = 0;
	int winner_trade = 0;
	int count = 0;
	int count_trade =0;
	float best = 0.0;
	std::vector<int> best_ensemble;
	std::vector<int> ensemble_best_runs;
	std::vector<int> ensemble_least_loser_runs;
	std::vector<int> ensemble_shortest_losing_streak;
	int best_runs = 0;
	int ensemble_n_runs = 0;

	float percent_correct = 0.000000000;
	int best_loser_runs_trade = labels.size();
	int loser_runs_count_trade=0;
	int loser_runs_trade = 0;

	int best_total_losing_streak = labels.size();
	int longest_total_losing_streak_trade =0;
	int longest_total_losing_streak =0;
	long double wl=0.000;
	//int wl=0;//wl++ for winner wl -- for losers
	int wl_trade=0;//wl++ for winner wl -- for losers
	boost::posix_time::ptime highest_wl_date;
	int highest_wl = 0;
	int highest_wl_trade = 0;
	boost::posix_time::ptime start_losing;
	boost::posix_time::ptime start_losing_trade;
	boost::posix_time::ptime start_losing1;
	boost::posix_time::ptime start_losing1_trade;
	boost::posix_time::ptime end_losing;
	boost::posix_time::ptime end_losing_trade;
	int highest_wl_when_losses_started_trade = 0;

	std::map<std::vector<int >, StatsTuple > potential_ensembles;
		std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles1;

	/*TODO:
	 * for std::vector<int> best_ensemble should be vector<std::vector<int>> to handle ties
	 * 	this needs to be done for everything that might be ties
	 * for losing runs need start date and end date
	 * need like a longest losing streak
	 * calculate PnL
	 */


	//index number in vector
	int price_index=0;
	int ud_index=1;
	int diff_index=2;


	long double total_pnl= 0.000000000000000;
	long double trade_entry_value = 0.0000000;
	long double trade_exit_value = 0.0000000000;
	long double trade_pnl = 0.0000000000;
	vector<long double > total_pnl_vect;
	vector<long double > trade_pnl_vect;

	long double peak_pnl =0.00000;
	long double trough_pnl = 0.0000;
	long double peak_pnl_when_losses_started = 0.0000000000;
	boost::posix_time::ptime max_winning_day_date;
	long double max_winning_day = 0.0000000000;
	boost::posix_time::ptime max_losing_day_date;
	long double max_losing_day = 0.0000000000;


	boost::posix_time::ptime startDate(boost::gregorian::date(2018,1,9),cBegin->second.begin()->first.time_of_day());
	boost::posix_time::ptime no_losses_in_this_year(boost::gregorian::date(2018,1,9),cBegin->second.begin()->first.time_of_day());
	boost::posix_time::ptime no_losses_in_this_year1(boost::gregorian::date(2019,1,9),cBegin->second.begin()->first.time_of_day());

	if(past_year == true){
		for(auto date_it = cBegin->second.begin(); date_it !=cBegin->second.end(); ++date_it)
			if(date_it->first.date() == startDate.date())
				startDate = date_it->first;
	}


	long double stat = 0;//statistic used to decided if ensemble should be put into the map of potential ensembles
	StatsTuple ensemble_stats;


	std::multimap<boost::posix_time::ptime, long>::iterator date_it;
	int ensemble_count = 0;

	//std::vector<int> want_ensemble = {0,14,18};

	//cout<<"past_year"<<past_year<<endl;
	//Total accuracy and number of runs for ensemble

	double tick_value = .25;
	for(; cBegin != cEnd; ++cBegin){
		//cout<<"AnalizeEnsembles ensemble "<<cBegin->first<<endl;

		if(past_year == true)
			date_it = cBegin->second.find(startDate);
		else
			date_it = cBegin->second.begin();

		//trade_entry_value = Sprd.find(date_it->first)->second[diff_index];//Sprd.find(cBegin->second.begin()->first)->second[diff_index];
		start_losing = date_it->first;//cBegin->second.begin()->first;
		start_losing_trade = date_it->first;//cBegin->second.begin()->first;

		start_losing = date_it->first;//cBegin->second.begin()->first;//put here on 01 22 2019, for when the very first trade is a loser
		highest_wl_date = date_it->first;//cBegin->second.begin()->first;

		//if(want_ensemble == cBegin->first)
		//	cout<<"break"<<endl;

		for(; date_it !=cBegin->second.end(); ++date_it){
			//for(auto date_it = cBegin->second.begin(); date_it !=cBegin->second.end(); ++date_it){
			//cout<<"labels Date"<<labels.find(date_it->first)->first<<endl;
			//ud = labels[date_it->first];



			ud = Sprd.find(date_it->first)->second[ud_index];

			if(date_it->second == 0)
				wl = -1* Sprd.find(date_it->first)->second[diff_index];
			else
				wl =  Sprd.find(date_it->first)->second[diff_index];

			/*if(ud == date_it->second)
				cout<<"wl should be positive "<<wl<<endl;
			else
				cout<<"wl should be negative "<<wl<<endl;*/

			//Per trade PnL
			if(prev_ud !=  date_it->second && date_it != cBegin->second.begin()){
				trade_pnl = total_pnl - trade_entry_value-tick_value;//cout<<"winner"<<endl;
				count_trade++;
				if(trade_pnl>0/*trade_entry_value > total_pnl*//*ud == date_it->second*/){
					winner_trade++;
					//for longest losing streak I want both number of losing trades in row and greatest losing pnl for a trade
					//longest losing streak
					if(loser_runs_count_trade>loser_runs_trade)
						loser_runs_trade = loser_runs_count_trade;
					loser_runs_count_trade = 0;
					wl_trade++;
					if(wl>highest_wl){
						highest_wl_trade = wl_trade;
						start_losing_trade = date_it->first;
					}
					//acc(percent_correct);
				}
				else{
					loser_runs_count_trade++;
					wl_trade--;
					if(std::abs(wl_trade - highest_wl_trade) > longest_total_losing_streak_trade){
						longest_total_losing_streak_trade = std::abs(wl_trade - highest_wl_trade);

						highest_wl_when_losses_started_trade= highest_wl_trade;
						start_losing1_trade = start_losing_trade;
						end_losing_trade = date_it->first;

					}
					//acc(percent_correct);
				}

				//cout<<"total_pnl "<<total_pnl<<" Percent Wining Trades "<<(float)winner_trade/(float)count_trade<<" peak_pnl "<<peak_pnl<<" trough_pnl "<<trough_pnl<<endl;

				trade_pnl_vect.push_back(trade_pnl);
				//cout<<"trade_pnl "<<trade_pnl<<endl;
				total_pnl += wl-tick_value;//Sprd.find(date_it->first)->second[diff_index];
				total_pnl_vect.push_back(total_pnl);

				trade_entry_value = total_pnl;

			}
			else{

				total_pnl += wl;//Sprd.find(date_it->first)->second[diff_index];
				total_pnl_vect.push_back(total_pnl);
			}

			//Total PnL/daily PnL
			if(wl > 0 /*Sprd.find(date_it->first)->second[diff_index]>0*/){
				if(wl > max_winning_day){
					max_winning_day =wl;
					max_winning_day_date = date_it->first;
				}
				//Peak to trough
				if(total_pnl > peak_pnl/*ud == date_it->second*/){
					//winner++;
					peak_pnl = total_pnl;
					//for longest losing streak I want both number of losing trades in row and greatest losing pnl for a trade
					//longest losing streak

					/*counts number of losers in a row if(loser_runs_count>loser_runs)
					loser_runs = loser_runs_count;
				loser_runs_count = 0;*/
					//wl++;
					/*if(wl>highest_wl)*///{
					//this highest_wl = wl; is replaced by this above peak_pnl = total_pnl;
					//{
					highest_wl_date = date_it->first;
					start_losing = date_it->first;
					//}
					//}
					//I don't reset this because I am looking for peak to trough trough_pnl = 0;//reset
				}
			}
			else{

				if(wl < max_losing_day){
					max_losing_day =wl;
					max_losing_day_date = date_it->first;
				}

				if(trough_pnl > total_pnl - peak_pnl){
					trough_pnl = total_pnl - peak_pnl;
					//longest_total_losing_streak = std::abs(wl - highest_wl);
					peak_pnl_when_losses_started = peak_pnl;
					//highest_wl_when_losses_started= highest_wl;
					start_losing1 = start_losing;
					end_losing = date_it->first;
				}
				//counts number of losers in a row loser_runs_count++;
				/*wl--;
				if(std::abs(wl - highest_wl) > longest_total_losing_streak){
					longest_total_losing_streak = std::abs(wl - highest_wl);
					if(!end_losing_streak){
						highest_wl_when_losses_started= highest_wl;
						start_losing1 = start_losing;
						end_losing = date_it->first;
						end_losing_streak = false;
					}
				}*/
			}

			prev_ud =  date_it->second;



		}
		//cout<<"cBegin->second.size() "<<cBegin->second.size()<<endl;
		//count = cBegin->second.size();
		count++;
		//if(total_pnl<0)
		//	cout<<"loser"<<endl;


		ensemble_count++;
		stat = total_pnl;//wl;
		percent_correct = (float)winner_trade/(float)count_trade;
		ensemble_n_runs = count_trade;
		//cout<<"Stat "<<stat<<" Percent Correct "<<percent_correct<<endl;

		if((no_losses_in_this_year.date().year() != end_losing.date().year() && no_losses_in_this_year1.date().year() != end_losing.date().year()
				&& past_year == false && percent_correct >= .49 && stat >= 1300) || (percent_correct >= .49 && stat >= 600 &&past_year == true )/*longest_total_losing_streak <= 20*/){
			if(potential_ensembles1.size() >= 30){

				/*ensemble_stats=std::make_tuple((float)winner/(float)count,ensemble_n_runs,acctual_n_runs,loser_runs,longest_total_losing_streak,
						start_losing1,end_losing,highest_wl,wl,highest_wl_when_losses_started,highest_wl_date);*/

				//uses total PnL ??
				ensemble_stats = std::make_tuple(percent_correct,ensemble_n_runs,acctual_n_runs,loser_runs_trade,trough_pnl,
						start_losing1,end_losing,peak_pnl,total_pnl,peak_pnl_when_losses_started,highest_wl_date,total_pnl_vect,trade_pnl_vect);

				PotentialEnesmbleCheck(stat, potential_ensembles1, cBegin->first,ensemble_stats);

			}
			else{

				//uses total PnL ??
				ensemble_stats = std::make_tuple(percent_correct,ensemble_n_runs,acctual_n_runs,loser_runs_trade,trough_pnl,
						start_losing1,end_losing,peak_pnl,total_pnl,peak_pnl_when_losses_started,highest_wl_date,total_pnl_vect,trade_pnl_vect);

				if(potential_ensembles1.find(stat)==potential_ensembles1.end()){
					std::map<vector<int>, StatsTuple > stat_map;
					stat_map[cBegin->first] = ensemble_stats;
					//potential_ensembles1.erase(potential_ensembles1.begin()->first);

					//potential_ensembles1.insert(std::pair<int, std::map<vector<int>, StatsTuple > >(stat, stat_map));
					potential_ensembles1[stat] = stat_map;
				}
				else
					potential_ensembles1[stat][cBegin->first] = ensemble_stats;


				/*cout<<"Total PnL "<<total_pnl<<" Ensemble n_runs "<<ensemble_n_runs<<" Acctual_n_runs "<<acctual_n_runs<<
													" best_loser_runs "<<loser_runs_trade<<" longest_total_losing_streak "<<longest_total_losing_streak<<
													" start_losing "<<start_losing1.date()<<" end_losing "<<end_losing.date()<<" highest_wl "<<highest_wl<<" wl "<<wl<<
													" "<<highest_wl_when_losses_started_trade<<" Ensemble "<<cBegin->first<<endl;*/
			}
		}


		/*if((float)winner_trade/(float)count_trade>=.5 && loser_runs_trade <= 5 && longest_total_losing_streak <= 5 && wl>=578){
			potential_ensembles[cBegin->first] = std::make_tuple((float)winner_trade/(float)count_trade,ensemble_n_runs,acctual_n_runs,loser_runs_trade,longest_total_losing_streak,
					start_losing1,end_losing,highest_wl,wl,highest_wl_when_losses_started_trade,max_winning_day_date);

			cout<<"Total PnL "<<total_pnl<<" Ensemble n_runs "<<ensemble_n_runs<<" Acctual_n_runs "<<acctual_n_runs<<
									" best_loser_runs "<<loser_runs_trade<<" longest_total_losing_streak "<<longest_total_losing_streak<<
									" start_losing "<<start_losing1.date()<<" end_losing "<<end_losing.date()<<" highest_wl "<<highest_wl<<" wl "<<wl<<
									" "<<highest_wl_when_losses_started_trade<<" Ensemble "<<cBegin->first<<endl;
			cout<<"Percent Correct "<<(float)winner/(float)count<<" Ensemble n_runs "<<ensemble_n_runs<<" Acctual_n_runs "<<acctual_n_runs<<
							" best_loser_runs "<<loser_runs<<" longest_total_losing_streak "<<longest_total_losing_streak<<
							" start_losing "<<start_losing1.date()<<" end_losing "<<end_losing.date()<<" highest_wl "<<highest_wl<<" wl "<<wl<<
							" "<<highest_wl_when_losses_started<<endl;
		}*/

		if(total_pnl>best){
			best = total_pnl;//(float)winner_trade/(float)count_trade;
			best_ensemble=cBegin->first;
		}
		if(std::abs(ensemble_n_runs-acctual_n_runs)<std::abs(best_runs-acctual_n_runs)){
			best_runs = ensemble_n_runs;
			ensemble_best_runs = cBegin->first;
		}
		if(loser_runs_trade<best_loser_runs_trade){
			best_loser_runs_trade = loser_runs_trade;
			ensemble_least_loser_runs = cBegin->first;
		}
		if(longest_total_losing_streak<best_total_losing_streak){
			best_total_losing_streak=longest_total_losing_streak;
			ensemble_shortest_losing_streak = cBegin->first;

		}

		//will need to store total_pnl_vect in another vector by ensemble
		trade_pnl = 0.000;
		trade_entry_value = 0.0;
		total_pnl_vect.clear();
		trade_pnl_vect.clear();
		wl_trade=0;
		count=0;
		ensemble_n_runs = 0;
		loser_runs_count_trade = 0;
		loser_runs_trade=0;
		longest_total_losing_streak =0;
		wl=0;
		highest_wl = 0;
		winner =0;
		total_pnl=0;
		winner_trade=0;
		count_trade=0;
		trough_pnl=0;
		peak_pnl = 0 ;
		acc = w_acc();
	}



	//cout<<"Best_ensemble "<<best_ensemble<<"_"<<best<<endl;
	//cout<<"best_runs_Ensemble "<< ensemble_best_runs<<"_"<<best_runs<<endl;
	//cout<<"Ensemble shortest losing streak "<<ensemble_least_loser_runs<<"_"<<best_loser_runs_trade<<endl;
	//cout<<"Ensemble shortest total losing streak "<<ensemble_shortest_losing_streak<<"_"<<best_total_losing_streak<<endl;

	return potential_ensembles1;

}


/*
 * @brief: builds and saves plot, also converts model numbers in the ensemble to a string so they can be used to name the plot
 *
 * @param string chart_path: just the path to were save the chart at
 */
void PlotAndSavePnL(vector<int> models_in_ensemble, std::map<vector<int>, StatsTuple > &ensemble, string chart_path){

	std::vector<std::pair<float, float>> data;
	std::vector<long double> converter = std::get<11>(ensemble.find(models_in_ensemble)->second);
	std::vector<float> values(converter.begin(), converter.end());

	string ensemble_name;
	for_each(models_in_ensemble.begin(),models_in_ensemble.end(),[&] (int i) {ensemble_name+=std::to_string(i)+"_";});
	ensemble_name.erase(--ensemble_name.end());

	{
		auto name = "simple";
		cvplot::setWindowTitle(name, "PnL");
		cvplot::moveWindow(name, 0, 0);
		cvplot::resizeWindow(name, 1000, 1000);
		auto &figure = cvplot::figure(name);
		figure.series("line")
	            				.setValue(values)
								.type(cvplot::Line)
								.color(cvplot::Black);
		figure.show(false);
		figure.save(chart_path,ensemble_name);
		figure.clear();

	}


	converter.clear();
	values.clear();
	converter = std::get<12>(ensemble.find(models_in_ensemble)->second);
	values.assign(converter.begin(), converter.end());

	ensemble_name.clear();
	for_each(models_in_ensemble.begin(),models_in_ensemble.end(),[&] (int i) {ensemble_name+=std::to_string(i)+"_";});
	//ensemble_name.erase(--ensemble_name.end());
	ensemble_name = ensemble_name + "PerTrade";

	{
		auto name = "simple";
		cvplot::setWindowTitle(name, "PnLPerTrade");
		cvplot::moveWindow(name, 0, 0);
		cvplot::resizeWindow(name, 1000, 1000);
		auto &figure = cvplot::figure(name);
		figure.series("line")
		            						.setValue(values)
											.type(cvplot::Line)
											.color(cvplot::Black);
		figure.show(false);
		figure.save(chart_path,ensemble_name);
		figure.clear();

	}


}

std::map<int, std::map<vector<int>, StatsTuple > > AnalizeEnsembles(vector<vector<int> > &combos, std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long> > &ensemble, string spread_filename, bool past_year ){


	//cout<<"AnalizeEnsembles"<<endl;
	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long> >::iterator cBegin = ensemble.begin();
	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long> >::iterator cEnd = ensemble.end();

	std::multimap<boost::posix_time::ptime, std::vector<long double> > Sprd;

	std::map<boost::posix_time::ptime, long > labels;
	MyData::SprdData(spread_filename, Sprd,true);
	//SprdData(caffe2::data_path+caffe2::spread_filename, Sprd);

	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator label_it;
	std::multimap<boost::posix_time::ptime, std::vector<long double >>::iterator Sprd_End = Sprd.end();

	long ud = 0;
	long prev_ud = 0;
	int acctual_n_runs = 0;

	//put labels into a map and count runs
	//and performs a check
	for(auto date_it = cBegin->second.begin(); date_it !=cBegin->second.end(); ++date_it){
		label_it = Sprd.find(date_it->first);
		//cout<<"cBegin->first "<<cBegin->first<<endl;
		if(label_it != Sprd_End){
			ud = label_it->second[0];
			labels[date_it->first]=ud;
			if(prev_ud != ud && date_it != cBegin->second.begin())
				acctual_n_runs++;
			prev_ud = ud;
			//cout<<"labels Date"<<labels.find(date_it->first)->first<<" ud "<<labels.find(date_it->first)->second<<endl;
		}
		else
			cout<<"Problem in AnalizeEnsembles Can't find date "<<label_it->first<<endl;
	}

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

	/*TODO:
	 * for std::vector<int> best_ensemble should be vector<std::vector<int>> to handle ties
	 * 	this needs to be done for everything that might be ties
	 * for losing runs need start date and end date
	 * need like a longest losing streak
	 * calculate PnL
	 */




	boost::posix_time::ptime startDate(boost::gregorian::date(2018,1,9),cBegin->second.begin()->first.time_of_day());

	if(past_year == true){
		for(auto date_it = cBegin->second.begin(); date_it !=cBegin->second.end(); ++date_it)
			if(date_it->first.date() == startDate.date())
				startDate = date_it->first;
	}


	int stat = 0;//statistic used to decided if ensemble should be put into the map of potential ensembles
	StatsTuple ensemble_stats;


	std::multimap<boost::posix_time::ptime, long>::iterator date_it;
	int ensemble_count = 0;
	//Total accuracy and number of runs for ensemble
	for(; cBegin != cEnd; ++cBegin){
		//cout<<"AnalizeEnsembles ensemble "<<cBegin->first<<endl;
		start_losing = cBegin->second.begin()->first;//put here on 01 22 2019, for when the very first trade is a loser
		highest_wl_date = cBegin->second.begin()->first;
		if(past_year == true)
			date_it = cBegin->second.find(startDate);
		else
			date_it = cBegin->second.begin();

		for(; date_it !=cBegin->second.end(); ++date_it){
		//for(auto date_it = cBegin->second.begin(); date_it !=cBegin->second.end(); ++date_it){
			//cout<<"labels Date"<<labels.find(date_it->first)->first<<" ud "<<labels[date_it->first]<<" pred "<<date_it->second<<endl;
			ud = labels[date_it->first];
			if(ud == date_it->second){
				winner++;
				//longest losing streak
				if(loser_runs_count>loser_runs)
					loser_runs = loser_runs_count;
				loser_runs_count = 0;
				wl++;
				//total_pnl_vect.push_back((long double)wl);
				if(wl>highest_wl){
					highest_wl = wl;
					highest_wl_date = date_it->first;
					/*if(end_losing_streak)*/{
						start_losing = date_it->first;
						end_losing_streak = true;
					}
				}
			}
			else{
				loser_runs_count++;
				wl--;
				//total_pnl_vect.push_back((long double)wl);
				if(std::abs(wl - highest_wl) > longest_total_losing_streak){
					longest_total_losing_streak = std::abs(wl - highest_wl);
					/*if(!end_losing_streak)*/{
						highest_wl_when_losses_started= highest_wl;
						start_losing1 = start_losing;
						end_losing = date_it->first;
						end_losing_streak = false;
					}
				}
			}
			if(prev_ud !=  date_it->second && date_it != cBegin->second.begin())
				ensemble_n_runs++;
			prev_ud =  date_it->second;
			count++;
		}
		//cout<<"cBegin->second.size() "<<cBegin->second.size()<<endl;
		//count = cBegin->second.size();

		//cout<<"Percent Correct "<<(float)winner/(float)count<<endl;

		percent_correct = (float)winner/(float)count;
		ensemble_count++;
		stat = wl;

		if(longest_total_losing_streak <= 20){
			if(potential_ensembles1.size() > 1500){

				ensemble_stats=std::make_tuple(percent_correct,ensemble_n_runs,acctual_n_runs,loser_runs,longest_total_losing_streak,
						start_losing1,end_losing,highest_wl,wl,highest_wl_when_losses_started,highest_wl_date,total_pnl_vect,trade_pnl_vect);

				PotentialEnesmbleCheck(stat, potential_ensembles1, cBegin->first,ensemble_stats);

			}
			else{
				ensemble_stats=std::make_tuple(percent_correct,ensemble_n_runs,acctual_n_runs,loser_runs,longest_total_losing_streak,
						start_losing1,end_losing,highest_wl,wl,highest_wl_when_losses_started,highest_wl_date,total_pnl_vect,trade_pnl_vect);

				if(potential_ensembles1.find(stat)==potential_ensembles1.end()){
					std::map<vector<int>, StatsTuple > stat_map;
					stat_map[cBegin->first] = ensemble_stats;
					//potential_ensembles1.erase(potential_ensembles1.begin()->first);

					//potential_ensembles1.insert(std::pair<int, std::map<vector<int>, StatsTuple > >(stat, stat_map));
					potential_ensembles1[stat] = stat_map;
				}
				else
					potential_ensembles1[stat][cBegin->first] = ensemble_stats;


				/*cout<<"Percent Correct "<<(float)winner/(float)count<<" Ensemble n_runs "<<ensemble_n_runs<<" Acctual_n_runs "<<acctual_n_runs<<
						" best_loser_runs "<<loser_runs<<" longest_total_losing_streak "<<longest_total_losing_streak<<
						" start_losing "<<start_losing1.date()<<" end_losing "<<end_losing.date()<<" highest_wl "<<highest_wl<<" wl "<<wl<<
						" "<<highest_wl_when_losses_started<<" Ensemble "<<cBegin->first<<endl;*/
			}
		}

		if((float)winner/(float)count>=.83 && loser_runs <= 3 && longest_total_losing_streak <= 4 /*&& wl>=578*/){
			potential_ensembles[cBegin->first] = std::make_tuple((float)winner/(float)count,ensemble_n_runs,acctual_n_runs,loser_runs,longest_total_losing_streak,
					start_losing1,end_losing,highest_wl,wl,highest_wl_when_losses_started,highest_wl_date,total_pnl_vect,trade_pnl_vect);
			cout<<"Percent Correct "<<(float)winner/(float)count<<" Ensemble n_runs "<<ensemble_n_runs<<" Acctual_n_runs "<<acctual_n_runs<<
					" best_loser_runs "<<loser_runs<<" longest_total_losing_streak "<<longest_total_losing_streak<<
					" start_losing "<<start_losing1.date()<<" end_losing "<<end_losing.date()<<" highest_wl "<<highest_wl<<" wl "<<wl<<
					" "<<highest_wl_when_losses_started<<" Ensemble "<<cBegin->first<<endl;
		}

		if((float)winner/(float)count>best){
			best = (float)winner/(float)count;
			best_ensemble=cBegin->first;
		}
		if(std::abs(ensemble_n_runs-acctual_n_runs)<std::abs(best_runs-acctual_n_runs)){
			best_runs = ensemble_n_runs;
			 ensemble_best_runs = cBegin->first;
		}
		if(loser_runs<best_loser_runs){
			best_loser_runs = loser_runs;
			ensemble_least_loser_runs = cBegin->first;
		}
		if(longest_total_losing_streak<best_total_losing_streak){
			best_total_losing_streak=longest_total_losing_streak;
			ensemble_shortest_losing_streak = cBegin->first;

		}

		count = 0;
		ensemble_n_runs = 0;
		loser_runs_count = 0;
		loser_runs=0;
		longest_total_losing_streak =0;
		wl=0;
		highest_wl = 0;
		winner =0;
	}


	past_year = true;//just to prevent it from doing the below
	if(past_year == false){




		std::vector<int> sample_intervals = {0,50,100,150,200,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,608};
		//std::vector<int> sample_intervals = {0,50,100,150,200,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750,775,800,825,850};
		//std::vector<int> sample_intervals = {0,50,100,150,200,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750,775,800,825,
		//		805,875,901};
		std::vector<int>::iterator sample_intervals_it = sample_intervals.begin();

		int sample_amount = 30;

		//std::multimap<boost::posix_time::ptime, long >::iterator date_it;
		std::multimap<boost::posix_time::ptime, long >::iterator start_it;
		std::multimap<boost::posix_time::ptime, long>::iterator end_it;
		string stat_func;
		double stat_result = 0.0;
		double percent_correct = 0.0;
		double best_range = 1;
		vector<int > best_range_model;
		count = 0;

		double stat_min=0.0;

		for(auto pe_it = potential_ensembles.begin(); pe_it != potential_ensembles.end(); ++pe_it){
			//for(cBegin = ensemble.begin(); cBegin != cEnd; ++cBegin){
			//cout<<"AnalizeEnsembles ensemble "<<cBegin->first<<endl;
			//cout<<"AnalizeEnsembles ensemble "<<pe_it->first<<endl;

			for(sample_intervals_it = sample_intervals.begin();sample_intervals_it!= sample_intervals.end(); ++sample_intervals_it){
				//date_it = cBegin->second.begin();
				date_it = ensemble.find(pe_it->first)->second.begin();
				std::advance(date_it,*sample_intervals_it);
				//cout<<"Samp Start Date "<<date_it->first;
				end_it = date_it;

				std::advance(end_it,sample_amount);

				if(std::distance(end_it, ensemble.find(pe_it->first)->second.end()) <= 0)
					//if(std::distance(end_it, cBegin->second.end()) <= 0)
					break;
				for(;  date_it !=end_it; ++date_it){
					//cout<<"labels Date"<<labels.find(date_it->first)->first<<endl;
					ud = labels[date_it->first];
					if(ud == date_it->second)
						winner++;
					count++;
				}

				percent_correct = (double)winner/(double)count;
				//cout<<" End Date "<<end_it->first<<" percent correct "<<percent_correct<<endl;
				acc(percent_correct);
				winner =0;
				count =0;
			}

			stat_min = stats["min"]();

			if(stat_min >=.53){
				cout<<"AnalizeEnsembles ensemble "<<pe_it->first<<endl;
				cout<<"Percent Correct "<<std::get<0>(potential_ensembles[pe_it->first])<<" Ensemble n_runs "<<std::get<1>(potential_ensembles[pe_it->first])<<" Acctual_n_runs "<<std::get<2>(potential_ensembles[pe_it->first])<<
						" best_loser_runs "<<std::get<3>(potential_ensembles[pe_it->first])<<" longest_total_losing_streak "<<std::get<4>(potential_ensembles[pe_it->first])<<
						" start_losing "<<std::get<5>(potential_ensembles[pe_it->first]).date()<<" end_losing "<<std::get<6>(potential_ensembles[pe_it->first]).date()<<" highest_wl "
						<<std::get<7>(potential_ensembles[pe_it->first])<<" wl "<<std::get<8>(potential_ensembles[pe_it->first])<<
						" "<<std::get<9>(potential_ensembles[pe_it->first])<<" Date Highest WL "<<std::get<9>(potential_ensembles[pe_it->first])<<endl;
				for (auto s : stats){
					stat_func = s.first;
					stat_result = s.second();
					cout<<" "<<stat_func<<" "<<stat_result;
				}
				cout<<endl;
			}
			if(stats["range"]()<best_range){
				best_range = stats["range"]();
				best_range_model = pe_it->first;
				//best_range_model = cBegin->first;
			}
			//stats.clear();
			//bacc::accumulator_set< double, bacc::stats<bacc::tag::min,bacc::tag::max,bacc::tag::mean,bacc::tag::median,
			//	            bacc::tag::variance > > acc;

			acc = w_acc();
		}
	}

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
//this gets the labels	this shoule be a function parameter 	SprdData(caffe2::data_path+caffe2::spread_filename, Sprd,false);

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

template<typename T>
void BuildEnsemble(T &all_pred_fnl){



	string spread_filename = "need to set this some how";

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


		potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble,spread_filename, false);
		//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
		potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,spread_filename, false);
		potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,spread_filename, true);

		FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_PnL_past_year,potential_ensembles_PnL_all_years);
		//FindIntersectingEnsemblesMap(potential_ensembles_PnL_all_years,potential_ensembles_PnL_past_year,potential_ensembles_all_years);

		//FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_past_year,potential_ensembles_PnL_all_years);

		std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long > > ensemble;

	}

	std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_all_years;
	std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_past_year;
	std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_all_years;
	std::map<int, std::map<vector<int>, StatsTuple > > potential_ensembles_PnL_past_year;

	potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, false);
	//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
	potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,false);
	potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,spread_filename, true);

	FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_PnL_past_year,potential_ensembles_PnL_all_years);
	//FindIntersectingEnsemblesMap(potential_ensembles_PnL_all_years,potential_ensembles_PnL_past_year,potential_ensembles_all_years);

	//FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_past_year,potential_ensembles_PnL_all_years);





}

template <typename T>
void BuildEnsembleThreading(T &all_pred_fnl){



	//PredictEnsemble();



	cout<<all_pred_fnl.begin()->second.size()<<endl;
	std::vector<int > n_models(all_pred_fnl.begin()->second.size());
	//std::generate (n_models.begin(), n_models.end(), UniqueNumber);
	for(uint i =0; i<n_models.size();++i)
		n_models[i] =i;

	vector<vector<int> > combos = Combinations(n_models,11);
	//std::vector<int> my_ensemble = {3,4,5,6,8,13,14,15,18,23,24};//was buy for
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
	//combos.push_back(my_ensemble);




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
				threads.push_back(std::thread(GoThroughCombosBinary,start, end, std::ref(combos)));
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

template<typename T>
void GoThroughCombos(int start, int end, vector<vector<int> > &combos, T &all_pred_fnl){



	std::multimap<boost::posix_time::ptime, std::vector<long>>::iterator cBegin = all_pred_fnl.begin();
	std::multimap<boost::posix_time::ptime, std::vector<long>>::iterator cEnd = all_pred_fnl.end();

	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long > > ensemble;
	std::multimap<boost::posix_time::ptime, long> ensemble_tmp;
	int count0 = 0;
	int count1 = 0;


	string spread_filename = "need to set this some how";

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
		{potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, false);
		//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
		potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,spread_filename,false);
		potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,spread_filename,true);

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

		potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, false);
		//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
		potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,spread_filename,false);
		potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,spread_filename,true);

		FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_PnL_past_year,potential_ensembles_PnL_all_years);
		//FindIntersectingEnsemblesMap(potential_ensembles_PnL_all_years,potential_ensembles_PnL_past_year,potential_ensembles_all_years);

		//FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_past_year,potential_ensembles_PnL_all_years);

}

template<typename T>
void GoThroughCombosBinary(int start, int end, vector<vector<int> > &combos, T &all_pred_fnl){


	std::multimap<boost::posix_time::ptime, std::vector<long>>::iterator cBegin = all_pred_fnl.begin();
	std::multimap<boost::posix_time::ptime, std::vector<long>>::iterator cEnd = all_pred_fnl.end();

	std::map<std::vector<int >, std::multimap<boost::posix_time::ptime, long > > ensemble;
	std::multimap<boost::posix_time::ptime, long> ensemble_tmp;
	int count0 = 0;
	int count1 = 0;

	string spread_filename = "need to set this some how";



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
		{potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, false);
		//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
		potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,spread_filename,false);
		potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,spread_filename,true);

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

		potential_ensembles_all_years = AnalizeEnsembles(combos, ensemble, false);
		//potential_ensembles_past_year = AnalizeEnsembles(combos, ensemble, true);
		potential_ensembles_PnL_all_years=AnalizeEnsemblesPnL(combos, ensemble,spread_filename,false);
		potential_ensembles_PnL_past_year=AnalizeEnsemblesPnL(combos, ensemble,spread_filename,true);

		FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_PnL_past_year,potential_ensembles_PnL_all_years);
		//FindIntersectingEnsemblesMap(potential_ensembles_PnL_all_years,potential_ensembles_PnL_past_year,potential_ensembles_all_years);

		//FindIntersectingEnsemblesMap(potential_ensembles_all_years,potential_ensembles_past_year,potential_ensembles_PnL_all_years);

}


namespace CPCO{
template <typename T>
void ReadConfig(string path_file_name, T ens){

	cout<<"CPCO"<<endl;
	int count = 0;
	//std::vector<std::string> filename;
	//filename=iterdir(file_path);

	//Search for the file of given name
	//string file_name=fileName;//"OFER_VDA_BMF_20170814.TXT";
	//const char *name = file_name.c_str();
	{
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
		data.push_back(E);


	auto FnlData = BuildFnlData(data);

	FnlDataToStruct(FnlData, ens.TRTEVA);

	Results.allBuys = std::accumulate(AllData.Labels.begin(), AllData.Labels.end(), 0);
	Results.allBuys = Results.allBuys/(AllData.Labels.size()-1);

	ens.n_features = FnlData.begin()->second.size()-1;

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
	Ensemble ens;
	for(auto iter : model_names){
		ReadConfig(iter, ens);
		TestModel(ens);

	}


	//will hhave to make sure I clear vectors appropriatels
	//will need to add Spolit data and I don't know if ens is correct for the below
	BuildAllPredVectFnl(ens);

}

}

#endif
