/*
 * Clouds.cpp
 *
 *  Created on: Oct 7, 2019
 *      Author: ryan
 */

#include <Clouds.h>




C10_DEFINE_int(iters_clouds, 4000, "The of training runs.");






Clouds::Clouds() {
	// TODO Auto-generated constructor stub

}

Clouds::~Clouds() {
	// TODO Auto-generated destructor stub
}

void Clouds::CloudsMain(){





	this->iterdir(train_imgs_path_, this->train_imgs_filenames_);

	MyImages1 my_images;
	this->ParseTrainCSV(this->main_path_+"train.csv", my_images);





	MyData::SplitData<std::vector<string>,std::unordered_map<string, std::vector<int> >, string, std::vector<vector<int> > > tet;
	this->GetImageData(my_images, tet.to_get);


	tet.BuildIters(my_images.image_data_vector_,my_images.labels_,this->id_label_seperated,my_images.decoded_pixels_,0,tet.split_data_only);


	//tet.BuildIters(my_images.decoded_pixels_,my_images.labels_,this->id_label_seperated,my_images.decoded_pixels_,0,tet.split_data_only);

		MyData::ModelConfig model_config;

		model_config.n_hide.push_back(1400*2100);
		model_config.n_hide.push_back(1400*2100);

		//this->BuildBaseNet(tet,model_config);

		this->ResNet50(my_images, tet,model_config);






}


template<typename T>
void Clouds::BuildBaseNet(T &data ,MyData::ModelConfig &model_config) {












	int n_features = 1400*2100;//data.train_features_[0].size();//this->base_features_[0].size();

	float base_learning_rate = -.0002998;
	int batch_size =2750;
	int classes = 1;

	std::cout << "Start training" << std::endl;
	string model_name = model_config.model_name;//this->model_name;
	string init_model_name = model_config.init_model_name();//this->init_model_name;

	string model_path = model_config.model_path;//this->file_path_;

	caffe2::NetDef init_model, predict_model;
			//caffe2::ModelUtil model(init_model, predict_model, model
			caffe2::ModelUtil model(init_model,predict_model);

			{

				caffe2::ResNetModel res_model(init_model, predict_model);

						res_model.Add(50, 4);

						res_model.predict.net.DebugString();
						cout<<"what"<<endl;






			}


	/*
flow is
1. input
2. FC
3. activation-> RELU, tanh, sigmoid, softmax
4. repeat 2 and 3 for total number of layers
5. cost funtion(measure of error rate, or the total loss over all the examples) loss function-> MSE RMSE,
	 */




	//set activation       model.predict.AddTanhOp("fourth_layer", "tanh3");
	//std::vector<string > layer({"1"});
	//vector<int > n_nodes_per_layer({8,16,8,2,3,2,3,5,1,2,4});
	string layer_name = " ";
	string activation = model_config.activation;//"LeakyRelu";//"Tanh";//
	string layer_in_name = " ";
	string layer_out_name = " ";

	model.predict.AddInput("input_blob");
	model.predict.AddInput(activation);
	model.predict.AddInput("target");
	//model.predict.AddInput("accuracy");
	model.predict.AddInput("loss");

	//Add layer, inputs are model to add, name of layer coming in, name of layer going out(i.e. name of this layer??)
	//number of neurons in this layer,  number of neurons in layer is connection to
	//think FC does add(matmul(inputs*w,b))

	model.predict.AddStopGradientOp("input_blob");


	for(int i =0; i< model_config.n_hide.size(); ++i){
		layer_name = std::to_string((i));



		if(i == 0){
			model.AddConvOps("input_blob","conv"+layer_name, n_features,model_config.n_hide[i], 2,3,7);
			model.AddSpatialBNOp("conv"+layer_name, "sbn"+layer_name, model_config.n_hide[i], .3,.3, false);
			//model.predict.AddScaleOp()
			//model.AddFcOps("input_blob", layer_name, n_features, model_config.n_hide[i]);
		}
		else{
			model.AddConvOps(activation+std::to_string(i),"conv"+layer_name, model_config.n_hide[i-1],model_config.n_hide[i], 2,3,7);
			model.AddSpatialBNOp("conv"+layer_name, "sbn"+layer_name, model_config.n_hide[i], .3,.3, false);
			//model.AddFcOps(activation+std::to_string(i),layer_name,model_config.n_hide[i-1], model_config.n_hide[i]);
		}
		if(activation == "LeakyRelu")
			model.predict.AddLeakyReluOp("sbn"+layer_name,activation+std::to_string(i+1),.3);//model.predict.AddSumOp(what, "sum");
		else if(activation == "Tanh")
			model.predict.AddTanhOp(layer_name,activation+std::to_string(i+1));

		//cout<<"layer_name "<<layer_name<<" activation+std::to_string(i+1) "<<activation+std::to_string(i+1)<<endl;

	}

	cout<<model.predict.net.DebugString()<<endl;
	//layer_name = activation+std::to_string(n_nodes_per_layer.size());
	layer_name = "last_layer";//"last_layer";//std::to_string(n_nodes_per_layer.size());
	//cout<<activation+std::to_string(n_nodes_per_layer.size())<<endl;
	//model.AddFcOps(activation+std::to_string(model_config.n_hide.size()),layer_name,model_config.n_hide[model_config.n_hide.size()-1], classes);

	//model.predict.AddConstantFillWithOp(1,"sum","loss");
	model.init.AddConstantFillOp({1},0.f,"loss");//model.predict.AddConstantFillOp({1},0.f,"loss");

	//had to add this so I could usev train.AddSgdOps();
	model.init.AddConstantFillOp({1},0.f,"one");//model.predict.AddConstantFillOp({1},0.f,"loss");

	//model.init.AddConstantFillWithOp(1.f, "loss", "loss_grad");
	//set loss
	//model.predict.AddSquaredL2Op(layer_name,"target","sql2");
	model.predict.AddL1DistanceOp(layer_name,"target","sql2");

	//model.predict.net.A
	model.predict.AddAveragedLossOp("sql2", "loss");



	model.AddIterOps();

	caffe2::NetDef f_int = model.init.net;
	caffe2::NetDef pred = model.predict.net;
	caffe2::ModelUtil save_model(f_int, pred, model_name);


	//cout<<model.predict.net.DebugString()<<endl;
	/*	caffe2::NetDef train_model(model.predict.net);
	caffe2::NetUtil train(train_model, "train");*/
	caffe2::NetDef train_init, train_predict;
	caffe2::ModelUtil train(train_init, train_predict,"train");
	string su = "relu";
	model.CopyTrain(layer_name, 1,train);

	//train.predict.AddInput("iter");//this should some how get done in void ModelUtil::CopyTrain(const std::string &layer, int out_size,ModelUtil &train)
	//train.predict.AddInput("one");//this should some how get done in void ModelUtil::CopyTrain(const std::string &layer, int out_size,ModelUtil &train)
	train.predict.AddConstantFillWithOp(1.f, "loss", "loss_grad");

//	//set optimizer
//	//model.AddAdamOps();
//	//model.AddRmsPropOps();
//	train.predict.AddGradientOps();
//	base_learning_rate = -1*base_learning_rate;
//	train.predict.AddLearningRateOp("iter","lr",base_learning_rate,.9);
//	train.AddSgdOps();
//	//train.AddRmsPropOps();
//
//
//	/*cout<<model.init.Proto()<<endl;
//	cout<<endl;
//	cout<<model.predict.Proto()<<endl;
//	cout<<endl;
//	cout<<train.init.Proto()<<endl;
//	cout<<endl;
//	cout<<train.predict.Proto()<<endl;*/
//	//Start training
//	caffe2::Workspace workspace("tmp");
//
//	// >>> log.debug("Training model")
//	std::cout << "Train model" << std::endl;
//
//	// >>> workspace.RunNetOnce(self.model.param_init_net)
//	CAFFE_ENFORCE(workspace.RunNetOnce(model.init.net));
//
//	auto epoch = 0;
//
//	// >>> CreateNetOnce(self.model.net)
//	workspace.CreateBlob("input_blob");
//	//workspace.CreateBlob("accuracy");
//	workspace.CreateBlob("loss");
//	workspace.CreateBlob("target");
//	//workspace.CreateBlob("one");
//
//	workspace.CreateBlob("lr");
//
//	CAFFE_ENFORCE(workspace.RunNetOnce(train.init.net));
//	CAFFE_ENFORCE(workspace.CreateNet(train.predict.net));//CAFFE_ENFORCE(workspace.CreateNet(train.net));
//
//	//cout<<train.init.net.name()<<" "<<model.init.net.name()<<endl;
//
//	CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));//CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));
//
//	float wFbefore = caffe2::BlobUtil(*workspace.GetBlob("2_w")).Get().data<float>()[0];
//	float wFafter = 0.00000000;
//	//	float wSefore = caffe2::BlobUtil(*workspace.GetBlob("1_w")).Get().data<float>()[0];
//	//float wLefore = BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0];
//
//	int nTrainBatches = batch_size;//FLAGS_batch;//TrainData.Features.size()/FLAGS_batch;
//
//	//CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));
//	std::vector<float> tmp;
//
//	//compute number of minibatches for training, validation and testing
//	int n_train_batches =data.train_features_.size()/ batch_size;//TrainData.Features.size() / batch_size;
//	int n_valid_batches = data.validate_features_.size()/ batch_size;// ValidateData.Features.size() / batch_size;
//	int n_test_batches = data.test_features_.size()/ batch_size;//TestData.Features.size() / batch_size;
//
//	//early-stopping parameters
//	int patience = 4000;//  # look as this many examples regardless
//	int patience_increase = 4;//  # wait this much longer when a new best is found
//	float improvement_threshold = 0.595; //a relative improvement of this much is
//	//considered significant
//	int validation_frequency = std::min(n_train_batches, patience / 2);
//
//	int iter= 0;
//
//	bool done_looping = false;
//	int server = 0;
//
//	std::vector<float> validation_losses;
//	float this_validation_loss =0.0;
//	float best_validation_loss = std::numeric_limits<float>::max();
//	int best_iter =0;
//	float test_score =0.0;
//
//	float train_score = 0.0;
//
//	vector<float>tmp_w(n_features);
//	vector<vector<float> > check_weights_w;
//
//	while((this->best_test_score > model_config.no_train_thresh)){
//		CAFFE_ENFORCE(workspace.RunNetOnce(train.init.net));
//		done_looping = false;
//		iter = 0;
//		patience = 4000;
//
//		std::vector<float> validation_losses;
//		this_validation_loss =0.0;
//		best_validation_loss = std::numeric_limits<float>::max();
//		best_iter =0;
//		test_score =0.0;
//
//		train_score = 0.0;
//		epoch = 0;
//		//while (epoch < n_epochs) and (not done_looping):
//		while (epoch < FLAGS_iters_clouds && !done_looping) {
//
//			epoch++;//this for like total number of iterations
//
//			//Train
//			for(auto minibatch_index = 0; minibatch_index<(n_train_batches); ++minibatch_index ){
//
//
//				{
//					std::vector<int> dim({nTrainBatches,data.train_features_[minibatch_index].size()});
//					caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, data.train_features_, minibatch_index, false);
//				}
//
//				{
//					std::vector<int> dim({nTrainBatches,1});
//					caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, data.train_labels_, minibatch_index, false);
//
//					//std::cout<<"Train Label "<<caffe2::BlobUtil(*workspace.GetBlob("target")).Get().DebugString()<<std::endl;
//
//				}
//
//
//				CAFFE_ENFORCE(workspace.RunNet(train.predict.net.name()));//CAFFE_ENFORCE(workspace.RunNet(train.net.name()));
//
//				/*			cout<<caffe2::BlobUtil(*workspace.GetBlob(layer_name)).Get().DebugString()<<endl;//data<float>()[0];
//			for(int i = 0; i<nTrainBatches; ++i)
//				cout<<"layr_name data "<<caffe2::BlobUtil(*workspace.GetBlob(layer_name)).Get().data<float>()[i]<<endl;
//
//			for(int i =0; i<n_features; ++i)
//					tmp_w[i] = caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[i];//cout<<caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[i]<<" ";
//				check_weights_w.push_back(tmp_w);
//				cout<<endl;*/
//				//cout<<"wFbefore "<<wFbefore <<" After "<< caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[0]<<endl;
//				//			cout<<"wSefore "<<wSefore<<" After "<< caffe2::BlobUtil(*workspace.GetBlob("1_w")).Get().data<float>()[0]<<endl;
//				//	cout<<"wLefore "<<wLefore<<" After "<< BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0]<<endl;
//
//
//
//
//				train_score = caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];// cout<<"Train Score "<<train_score<<endl;
//
//				iter = (epoch - 1) * n_train_batches + minibatch_index;
//
//				if((iter + 1) % validation_frequency == 0){
//
//					{
//						//std::vector<int> dim({nTrainBatches,ValidateData.Features[minibatch_index].size()});
//						std::vector<int> dim({nTrainBatches,data.validate_features_[minibatch_index].size()});
//						//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, ValidateData.Features, minibatch_index,false);
//						caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, data.validate_features_, minibatch_index,false);
//					}
//
//					{
//						std::vector<int> dim({nTrainBatches,1});
//						//BlobUtil(*workspace.CreateBlob("target")).Set(dim, ValidateData.Labels,minibatch_index, false);
//						caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, data.validate_labels_,minibatch_index, false);
//						//std::cout<<"Validate Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
//					}
//
//
//					// >>> workspace.RunNet(self.forward_net.Name())
//					CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));
//
//					//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;
//
//
//					//cout<<"Validate accuracy "<<1.0-BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0]<<endl;
//
//					//this_validation_loss =1.0-caffe2::BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
//					this_validation_loss =caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
//					//cout<<"Valied Score "<<this_validation_loss<<endl;// 1-(float)(nCorrect)/(float)(countValue);//sess.run([accuracy], feed_dict={X: batch_x, Y: batch_y})[0]#numpy.mean(validation_losses)
//					//cout<<"Validate Percent Correct "<<(float)(nCorrect)/(float)(countValue)<<endl;
//
//					validation_losses.push_back(this_validation_loss);
//					// if we got the best validation score until now
//					if(this_validation_loss < best_validation_loss){
//						//improve patience if loss improvement is good enough
//						if( this_validation_loss < best_validation_loss * improvement_threshold)
//							patience = max(patience, iter * patience_increase);
//
//						best_validation_loss = this_validation_loss;
//						best_iter = iter;
//
//						//cout<<"optimizer_iteration "<<BlobUtil(*workspace.GetBlob("optimizer_iteration")).Get() <<endl;
//
//						{
//							//std::vector<int> dim({nTrainBatches,TestData.Features[minibatch_index].size()});
//							std::vector<int> dim({nTrainBatches,data.test_features_[minibatch_index].size()});
//							//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, TestData.Features,minibatch_index, false);
//							caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, data.test_features_,minibatch_index, false);
//						}
//
//						{
//							std::vector<int> dim({nTrainBatches,1});
//							//BlobUtil(*workspace.CreateBlob("target")).Set(dim, TestData.Labels, minibatch_index, false);
//							caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, data.test_labels_, minibatch_index, false);
//							//std::cout<<"Test Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
//						}
//
//						// >>> workspace.RunNet(self.forward_net.Name())
//						CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));
//
//						//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;
//
//						test_score = caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
//						//cout<<"Test Score "<<test_score<<endl;//(float)(nCorrect)/(float)(countValue);
//
//						cout<<"Train "<<train_score<< " Validate "<<this_validation_loss<<" Test "<<test_score<<endl;
//
//						//cout<<"wFbefore "<<wFbefore*10000 <<" After "<< caffe2::BlobUtil(*workspace.GetBlob("2_w")).Get().data<float>()[0]*10000<<endl;
//						//cout<<"Test accuracy "<<1.0-BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0]<<endl;
//
//						caffe2::NetDef deploy_init_model;  // the final initialization model
//						caffe2::ModelUtil deploy(deploy_init_model, save_model.predict.net,model.init.net.name());
//						//caffe2::ModelUtil deploy(deploy_init_model, model.predict.net,model.init.net.name());
//
//
//						save_model.CopyDeploy(deploy, workspace);
//						//CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));
//						if(server == 0){
//							//cout<<"Save Model"<<endl;//mdir=saver.save(sess, save_path='/home/ryan/workspace/MultiLayerTen/'+modelName, global_step=tf.train.global_step(sess,global_step_tensor))
//							caffe2::WriteProtoToBinaryFile(deploy.init.net, model_path+init_model_name);//caffe2::WriteProtoToTextFile(deploy.init.net, model_path+init_model_name+".pbtxt");//
//							caffe2::WriteProtoToBinaryFile(model.predict.net, model_path+ model_name);//caffe2::WriteProtoToTextFile(model.predict.net, model_path+ model_name+".pbtxt");//
//
//						}
//						else{
//							//cout<<"Save Model"<<endl;//mdir=saver.save(sess, save_path='/home/riley/data/HE/HE1-2Models/'+modelName, global_step=tf.train.global_step(sess,global_step_tensor))
//							caffe2::WriteProtoToTextFile(model.init.net, "/home/ryan/workspace/adsf/initModel");
//							caffe2::WriteProtoToTextFile(model.predict.net, "/home/ryan/workspace/adsf/model1");
//						}
//						//save the best model
//
//					}
//
//				}
//
//
//				if(patience <= iter){
//					done_looping = true;
//					break;
//				}
//
//			}
//
//
//			//std::cout << "Smooth loss: " << smooth_loss << std::endl;
//			//std::cout<<"last_layer_w After Training "<<BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0]<<std::endl;
//			if((test_score+this_validation_loss)/2<this->best_test_score){
//				best_test_score = (test_score+this_validation_loss)/2;
//				cout<<"Best Test Score "<<best_test_score<<endl;
//			}
//
//		}
//	}
//
//
//	//cout<<"Optimization complete. Best validation score of "<<best_validation_loss<< " obtained at iteration "<<best_iter + 1<<
//	//		" with test performance "<<test_score<<endl;
//
//
//	//cout<<"wFbefore "<<wFbefore <<" After "<< BlobUtil(*workspace.GetBlob("first_layer_w")).Get().data<float>()[0]<<endl;
//	//cout<<"wSefore "<<wSefore<<" After "<< BlobUtil(*workspace.GetBlob("second_layer_w")).Get().data<float>()[0]<<endl;
//	//cout<<"wLefore "<<wLefore<<" After "<< BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0]<<endl;
//
//
//	//if(best_test_score > model_config.no_train_thresh){
//	//	cout<<"Best Test Score "<<best_test_score<<endl;
//	//	this->BuildBaseNet(data,model_config);
//	//}
}
template<typename T, typename T1>
void Clouds::ResNet50( T &data, T1 &data_it, MyData::ModelConfig &model_config){


	//caffe

	int n_features = 1400*2100;//data.train_features_[0].size();//this->base_features_[0].size();

	float base_learning_rate = -.0002998;
	int batch_size = 1;
	int classes = 1;

	std::cout << "Start training" << std::endl;
	string model_name = model_config.model_name;//this->model_name;
	string init_model_name = model_config.init_model_name();//this->init_model_name;

	string model_path = model_config.model_path;//this->file_path_;

	caffe2::NetDef init_model, predict_model;
			//caffe2::ModelUtil model(init_model, predict_model, model
			//caffe2::ModelUtil model(init_model,predict_model);



				caffe2::ResNetModel res_model(init_model, predict_model);

						res_model.Add(50, 4,true);
				//res_model.Add(5, 100,true);

						res_model.predict.net.DebugString();

						cout<<"what"<<endl;








			//caffe2::NetDef f_int = model.init.net;
			//	caffe2::NetDef pred = model.predict.net;
			//	caffe2::ModelUtil save_model(f_int, pred, model_name);


				//cout<<model.predict.net.DebugString()<<endl;
				/*	caffe2::NetDef train_model(model.predict.net);
				caffe2::NetUtil train(train_model, "train");*/
				caffe2::NetDef train_init, train_predict;
				caffe2::ModelUtil train(train_init, train_predict,"train");
				string su = "relu";
				//model.CopyTrain(layer_name, 1,train);


		caffe2::Workspace workspace("tmp");

		// >>> log.debug("Training model")
		std::cout << "Train model" << std::endl;

		// >>> workspace.RunNetOnce(self.model.param_init_net)
		//CAFFE_ENFORCE(workspace.RunNetOnce(model.init.net));

		//workspace.CreateBlob("data");
			workspace.CreateBlob("softmax");
			workspace.CreateBlob("label");

			res_model.init.Proto();

			CAFFE_ENFORCE(workspace.RunNetOnce(res_model.init.net));

		//	CAFFE_ENFORCE(workspace.CreateNet(res_model.init.net));


//			sometihing needs to output label so it can be inputed to the softmax

		CAFFE_ENFORCE(workspace.CreateNet(res_model.predict.net));

		auto epoch = 0;

		// >>> CreateNetOnce(self.model.net)
//		workspace.CreateBlob("data");//workspace.CreateBlob("input_blob");
		//workspace.CreateBlob("accuracy");
		//workspace.CreateBlob("loss");
		//workspace.CreateBlob("label");//workspace.CreateBlob("target");
		//workspace.CreateBlob("one");

		//workspace.CreateBlob("lr");

		//CAFFE_ENFORCE(workspace.RunNetOnce(train.init.net));
		//CAFFE_ENFORCE(workspace.CreateNet(train.predict.net));//CAFFE_ENFORCE(workspace.CreateNet(train.net));

		//cout<<train.init.net.name()<<" "<<model.init.net.name()<<endl;

		//CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));//CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));


		int nTrainBatches = batch_size;//FLAGS_batch;//TrainData.Features.size()/FLAGS_batch;

		//CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));
		std::vector<float> tmp;

		//compute number of minibatches for training, validation and testing
		int n_train_batches = data_it.train_labels_.size()/ batch_size;//data.image_data_vector_.size()/ batch_size;//TrainData.Features.size() / batch_size;
		int n_valid_batches =data_it.validate_labels_.size()/ batch_size;// data.image_data_vector_.size()/ batch_size;// ValidateData.Features.size() / batch_size;
		int n_test_batches = data_it.test_labels_.size()/ batch_size;//data.image_data_vector_.size()/ batch_size;//TestData.Features.size() / batch_size;

		//early-stopping parameters
		int patience = 4000;//  # look as this many examples regardless
		int patience_increase = 4;//  # wait this much longer when a new best is found
		float improvement_threshold = 0.595; //a relative improvement of this much is
		//considered significant
		int validation_frequency = std::min(n_train_batches, patience / 2);

		int iter= 0;

		bool done_looping = false;
		int server = 0;

		std::vector<float> validation_losses;
		float this_validation_loss =0.0;
		float best_validation_loss = std::numeric_limits<float>::max();
		int best_iter =0;
		float test_score =0.0;

		float train_score = 0.0;

		vector<float>tmp_w(n_features);
		vector<vector<float> > check_weights_w;
		string id = " ";
		string id_label = " ";
		cout<<"res_model.init.net.DebugString() ";
		res_model.init.net.DebugString();
		cout<<" end "<<endl;
		while((best_test_score > model_config.no_train_thresh)){
			CAFFE_ENFORCE(workspace.RunNetOnce(res_model.init.net));
			done_looping = false;
			//iter = 0;
			patience = 4000;

			std::vector<float> validation_losses;
			this_validation_loss =0.0;
			best_validation_loss = std::numeric_limits<float>::max();
			best_iter =0;
			test_score =0.0;

			train_score = 0.0;
			epoch = 0;
			//while (epoch < n_epochs) and (not done_looping):
			while (epoch < FLAGS_iters_clouds && !done_looping) {

				epoch++;//this for like total number of iterations

				//Train
				int minibatch_index = 0;
				for(auto minibatch_index = 0; minibatch_index<(n_train_batches); ++minibatch_index ){
				//for(data.labels_one_hot_it_ = data.labels_one_hot_.begin();data.labels_one_hot_it_ != data.labels_one_hot_.end(); ++data.labels_one_hot_it_,++minibatch_index ){


					id_label = data_it.train_features_[minibatch_index];
					//decoded_pixels_label_it_ = decoded_pixels_.find()
					id = id_label;//this->id_labels_to_id.find(id_label)->second;
					data.image_data_vector_it_ = data.image_data_vector_.find(id);
					//out<<"data.image_data_vector_ "<<data.image_data_vector_<<endl;

					cout<<"data.image_data_vector_.find(id)->second "<<data.image_data_vector_.find(id)->second<<endl;
					cout<<"data.image_data_vector_it_->second "<<data.image_data_vector_it_->second<<endl;
					{
						//std::vector<int> dim({nTrainBatches,data.image_data_vector_it_->second.size()});
						int size = 224*224*3;
						std::vector<int> dim({1,3,224,224});//std::vector<int> dim({1,3,224,224});
						caffe2::BlobUtil(*workspace.CreateBlob("data")).Set(size, dim, data.image_data_vector_it_->second, false);
					}

					//data.labels_one_hot_it_ = data.labels_one_hot_.find(id_label);
					//data.labels_one_hot_it_ = data.labels_one_hot_.find(id_label);



					{
						std::vector<int> dim({1});
						caffe2::BlobUtil(*workspace.CreateBlob("label")).Set(dim,data_it.train_labels_.find(id_label)->second, 0, false);

						//std::vector<int> dim({1});
						//caffe2::BlobUtil(*workspace.CreateBlob("label")).Set(dim,data_it.train_labels_.find(id_label)->second, minibatch_index, false);

						//std::cout<<"Train Label "<<caffe2::BlobUtil(*workspace.GetBlob("target")).Get().DebugString()<<std::endl;

					}




					cout<<caffe2::BlobUtil(*workspace.GetBlob("label")).Get().data<int>()[0]<<endl;

					CAFFE_ENFORCE(workspace.RunNet(res_model.predict.net.name()));//CAFFE_ENFORCE(workspace.RunNet(train.net.name()));
					cout<<caffe2::BlobUtil(*workspace.GetBlob("final_avg")).Get().DebugString()<<endl;

					cout<<"res_model.init.net.DebugString() ";
							res_model.init.net.DebugString();
							cout<<" end "<<endl;


					/*			cout<<caffe2::BlobUtil(*workspace.GetBlob(layer_name)).Get().DebugString()<<endl;//data<float>()[0];
				for(int i = 0; i<nTrainBatches; ++i)
					cout<<"layr_name data "<<caffe2::BlobUtil(*workspace.GetBlob(layer_name)).Get().data<float>()[i]<<endl;

				for(int i =0; i<n_features; ++i)
						tmp_w[i] = caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[i];//cout<<caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[i]<<" ";
					check_weights_w.push_back(tmp_w);
					cout<<endl;*/
					//cout<<"wFbefore "<<wFbefore <<" After "<< caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[0]<<endl;
					//			cout<<"wSefore "<<wSefore<<" After "<< caffe2::BlobUtil(*workspace.GetBlob("1_w")).Get().data<float>()[0]<<endl;
					//	cout<<"wLefore "<<wLefore<<" After "<< BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0]<<endl;


							cout<<caffe2::BlobUtil(*workspace.GetBlob("data")).Get().data<float>()[0]<<endl;
							cout<<caffe2::BlobUtil(*workspace.GetBlob("data")).Get().data<float>()[1]<<endl;
							cout<<caffe2::BlobUtil(*workspace.GetBlob("data")).Get().data<float>()[2]<<endl;
					cout<<caffe2::BlobUtil(*workspace.GetBlob("data")).Get().data<float>()[3]<<endl;
					cout<<caffe2::BlobUtil(*workspace.GetBlob("data")).Get().data<float>()[4]<<endl;



					train_score = caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];

					cout<<"Train Score "<<train_score<<endl;


					cout<<caffe2::BlobUtil(*workspace.GetBlob("softmax")).Get().data<float>()[0]<<endl;
					cout<<caffe2::BlobUtil(*workspace.GetBlob("softmax")).Get().data<float>()[1]<<endl;
					cout<<caffe2::BlobUtil(*workspace.GetBlob("softmax")).Get().data<float>()[2]<<endl;
					cout<<caffe2::BlobUtil(*workspace.GetBlob("softmax")).Get().data<float>()[3]<<endl;

					cout<<caffe2::BlobUtil(*workspace.GetBlob("label")).Get().data<int>()[0]<<endl;

					cout<<caffe2::BlobUtil(*workspace.GetBlob("pred")).Get().data<float>()[0]<<endl;
					cout<<caffe2::BlobUtil(*workspace.GetBlob("argmax")).Get().data<long>()[0]<<endl;


					if(caffe2::BlobUtil(*workspace.GetBlob("pred")).Get().data<float>()[0] >1)
						cout<<"what"<<endl;

					cout<<"comp_15_sum_3 "<<caffe2::BlobUtil(*workspace.GetBlob("comp_15_sum_3")).Get().DebugString()<<endl;
					//cout<<caffe2::BlobUtil(*workspace.GetBlob("top-5")).Get().data<float>()[0]<<endl;
					//cout<<caffe2::BlobUtil(*workspace.GetBlob("top-5")).Get().data<float>()[1]<<endl;
					//cout<<caffe2::BlobUtil(*workspace.GetBlob("top-5")).Get().data<float>()[2]<<endl;
					//cout<<caffe2::BlobUtil(*workspace.GetBlob("top-5")).Get().data<float>()[3]<<endl;




					iter = (epoch - 1) * n_train_batches + minibatch_index;

					/*if((iter + 1) % validation_frequency == 0){

						{
							//std::vector<int> dim({nTrainBatches,ValidateData.Features[minibatch_index].size()});
							std::vector<int> dim({nTrainBatches,data.validate_features_[minibatch_index].size()});
							//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, ValidateData.Features, minibatch_index,false);
							caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, data.validate_features_, minibatch_index,false);
						}

						{
							std::vector<int> dim({nTrainBatches,1});
							//BlobUtil(*workspace.CreateBlob("target")).Set(dim, ValidateData.Labels,minibatch_index, false);
							caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, data.validate_labels_,minibatch_index, false);
							//std::cout<<"Validate Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
						}


						// >>> workspace.RunNet(self.forward_net.Name())
						CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));

						//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;


						//cout<<"Validate accuracy "<<1.0-BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0]<<endl;

						//this_validation_loss =1.0-caffe2::BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
						this_validation_loss =caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
						//cout<<"Valied Score "<<this_validation_loss<<endl;// 1-(float)(nCorrect)/(float)(countValue);//sess.run([accuracy], feed_dict={X: batch_x, Y: batch_y})[0]#numpy.mean(validation_losses)
						//cout<<"Validate Percent Correct "<<(float)(nCorrect)/(float)(countValue)<<endl;

						validation_losses.push_back(this_validation_loss);
						// if we got the best validation score until now
						if(this_validation_loss < best_validation_loss){
							//improve patience if loss improvement is good enough
							if( this_validation_loss < best_validation_loss * improvement_threshold)
								patience = max(patience, iter * patience_increase);

							best_validation_loss = this_validation_loss;
							best_iter = iter;

							//cout<<"optimizer_iteration "<<BlobUtil(*workspace.GetBlob("optimizer_iteration")).Get() <<endl;

							{
								//std::vector<int> dim({nTrainBatches,TestData.Features[minibatch_index].size()});
								std::vector<int> dim({nTrainBatches,data.test_features_[minibatch_index].size()});
								//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, TestData.Features,minibatch_index, false);
								caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, data.test_features_,minibatch_index, false);
							}

							{
								std::vector<int> dim({nTrainBatches,1});
								//BlobUtil(*workspace.CreateBlob("target")).Set(dim, TestData.Labels, minibatch_index, false);
								caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, data.test_labels_, minibatch_index, false);
								//std::cout<<"Test Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
							}

							// >>> workspace.RunNet(self.forward_net.Name())
							CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));

							//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;

							test_score = caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
							//cout<<"Test Score "<<test_score<<endl;//(float)(nCorrect)/(float)(countValue);

							cout<<"Train "<<train_score<< " Validate "<<this_validation_loss<<" Test "<<test_score<<endl;

							//cout<<"wFbefore "<<wFbefore*10000 <<" After "<< caffe2::BlobUtil(*workspace.GetBlob("2_w")).Get().data<float>()[0]*10000<<endl;
							//cout<<"Test accuracy "<<1.0-BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0]<<endl;

							caffe2::NetDef deploy_init_model;  // the final initialization model
							caffe2::ModelUtil deploy(deploy_init_model, save_model.predict.net,model.init.net.name());
							//caffe2::ModelUtil deploy(deploy_init_model, model.predict.net,model.init.net.name());


							save_model.CopyDeploy(deploy, workspace);
							//CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));
							if(server == 0){
								//cout<<"Save Model"<<endl;//mdir=saver.save(sess, save_path='/home/ryan/workspace/MultiLayerTen/'+modelName, global_step=tf.train.global_step(sess,global_step_tensor))
								caffe2::WriteProtoToBinaryFile(deploy.init.net, model_path+init_model_name);//caffe2::WriteProtoToTextFile(deploy.init.net, model_path+init_model_name+".pbtxt");//
								caffe2::WriteProtoToBinaryFile(model.predict.net, model_path+ model_name);//caffe2::WriteProtoToTextFile(model.predict.net, model_path+ model_name+".pbtxt");//

							}
							else{
								//cout<<"Save Model"<<endl;//mdir=saver.save(sess, save_path='/home/riley/data/HE/HE1-2Models/'+modelName, global_step=tf.train.global_step(sess,global_step_tensor))
								caffe2::WriteProtoToTextFile(model.init.net, "/home/ryan/workspace/adsf/initModel");
								caffe2::WriteProtoToTextFile(model.predict.net, "/home/ryan/workspace/adsf/model1");
							}
							//save the best model

						}

					}*/


					if(patience <= iter){
						done_looping = true;
						break;
					}

				}


				//std::cout << "Smooth loss: " << smooth_loss << std::endl;
				//std::cout<<"last_layer_w After Training "<<BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0]<<std::endl;
				if((test_score+this_validation_loss)/2<this->best_test_score){
					best_test_score = (test_score+this_validation_loss)/2;
					cout<<"Best Test Score "<<best_test_score<<endl;
				}

			}
		}




}

/*
 * @brief: performs run length decoding
 * @param: MyImages &my_images object that holds the image data
 * @param: string image_id the id of the image we are decoding
 * @param: bool display_mask to display the mask on the image or not default is false
 *
 */
void Clouds::RLEdecode( MyImages1 &my_images, string image_id, vector<string> &to_get, bool display_mask){

	int start = 0;
	int color = 278;
	int row = 0;
	int col = 0;
	int count = 0;
	int n_pixel = 0;
	int max_j_row = 0;
	vector<int> deocded_tmp;//
	uchar fu = '278';
	//cout<<"fu "<< fu<<" float)fu "<<(float)fu<<endl;
	//cout<<"(float)color "<<(float)color<<endl;

	cv::Mat image_data = my_images.image_data_.find(image_id)->second;
	int n_channels = image_data.channels();


	//cout<<my_images.encoded_pixels_id_it_->first<<endl;

	cout<<"shape: cols "<<image_data.cols<<" rows "<<image_data.rows<<" step "<<image_data.step<< endl;

	//my_images.encoded_pixels_label_id_  = my_images.encoded_pixels_id_it_->second.begin();
	//my_images.decoded_pixels_label_id_  = my_images.decoded_pixels_id_it_->second.begin();
	this->train_id_labels_names_it_ = this->train_id_labels_names_.find(image_id);
	cout<<this->train_id_labels_names_it_->first<<endl;

	for(auto labels = this->train_id_labels_names_it_->second.begin(); labels != this->train_id_labels_names_it_->second.end(); ++labels, ++count){
		cout<<*labels<<endl;
		to_get.push_back(*labels);
		my_images.encoded_pixels_label_it_ = my_images.encoded_pixels_.find(*labels);
		my_images.decoded_pixels_label_it_ = my_images.decoded_pixels_.find(*labels);



		//cout<<my_images.decoded_pixels_label_it_->second<<endl;
		my_images.decoded_pixels_label_it_->second.resize(my_images.decoded_pixels_label_it_->second[0]);
		for(uint k = 0; k<my_images.encoded_pixels_label_it_->second.size(); ++k){
			start = my_images.encoded_pixels_label_it_->second[k][my_images.start_pixel_];
			my_images.decoded_pixels_label_it_->second[n_pixel] = start;
			++n_pixel;
			col = std::floor(start/image_data.rows);

			row = start - col*image_data.rows;
			//cout<<start/image_data.rows*image_data.rows<<endl;

			if(row < image_data.rows){
			for(int j = 0; j<my_images.encoded_pixels_label_it_->second[k][my_images.num_pixels_]; ++j){
				if(j+row >= image_data.rows)
					break;
				my_images.decoded_pixels_label_it_->second[n_pixel] = start+j;
				++n_pixel;

				//if(display_mask == true){
					uchar *peg_ptr = image_data.ptr<uchar>(j+row);
					//if(max_j_row < j+row){
					//	max_j_row = j+row;
					//cout<<"max_j_row "<<max_j_row<<endl;
					//}
					//string peg_string(reinterpret_cast<char *>(peg_ptr));
					//cout<<sizeof(peg_ptr)<<" peg_ptr "<<peg_string<<endl;
					peg_ptr[col*n_channels+1] =  (uchar)color;
					/*for(int i = 0; i < image_data.total()*image_data.channels();i+=3){
					if(image_data.data[i+1] == (uchar)color)
						cout<<"i "<< i<<" "<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[i+1]<<" fu "<<float(fu)<<endl;
				}
				cout<<"row*3+col*n_channels "<<row*3+col*n_channels<<" "<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*3+col]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*3+col+1]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*3+col+2]<<endl;

				cout<<"row*col*n_channels "<<row*col*n_channels<<" "<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*col*n_channels]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*col*n_channels+1]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*col*n_channels+2]<<endl;*/

				//}
			}
			}

		}
		n_pixel = 0;
		//break;

	}

	cout<<"after all of shape: cols "<<image_data.cols<<" rows "<<image_data.rows<<" step "<<image_data.step<< endl;

	if(display_mask == true){
		cv::imshow(image_id, image_data);

		//cv::imwrite((image_id+".jpg"),image_data);

		cv::waitKey(0);
		cv::destroyWindow(image_id);
	}

	//why is there so many in this my_images.decoded_pixels_.find(*labels)
	//there should only be like 10 * labels per id



}


/*
 * @brief: performs run length decoding applys the mask and resizes
 *
 * @param: MyImages &my_images object that holds the image data
 * @param: string image_id the id of the image we are decoding
 * @param: bool display_mask to display the mask on the image or not default is false
 *
 */
void Clouds::ApplyMaskAndRezise( MyImages1 &my_images, string image_id, vector<string> &to_get, cv::Mat &jpegdat, vector<int> &new_shape, bool display_mask){

	int start = 0;
	int color = 278;
	int row = 0;
	int col = 0;
	int count = 0;
	int n_pixel = 0;
	int max_j_row = 0;
	vector<int> deocded_tmp;//
	std::vector<float> contents;

	uchar fu = '278';
	//cout<<"fu "<< fu<<" float)fu "<<(float)fu<<endl;
	//cout<<"(float)color "<<(float)color<<endl;

	//cv::Mat image_data; //=my_images.image_data_.find(image_id)->second;
	int n_channels = jpegdat.channels();//image_data.channels();





	//cout<<my_images.encoded_pixels_id_it_->first<<endl;

	//cout<<"shape: cols "<<image_data.cols<<" rows "<<image_data.rows<<" step "<<image_data.step<< endl;

	//my_images.encoded_pixels_label_id_  = my_images.encoded_pixels_id_it_->second.begin();
	//my_images.decoded_pixels_label_id_  = my_images.decoded_pixels_id_it_->second.begin();
	this->train_id_labels_names_it_ = this->train_id_labels_names_.find(image_id);
	cout<<this->train_id_labels_names_it_->first<<endl;

	for(auto labels = this->train_id_labels_names_it_->second.begin(); labels != this->train_id_labels_names_it_->second.end(); ++labels, ++count){

		cv::Mat image_data = jpegdat.clone();

		cout<<*labels<<endl;
		to_get.push_back(*labels);
		my_images.encoded_pixels_label_it_ = my_images.encoded_pixels_.find(*labels);
		my_images.decoded_pixels_label_it_ = my_images.decoded_pixels_.find(*labels);



		//cout<<my_images.decoded_pixels_label_it_->second<<endl;
		my_images.decoded_pixels_label_it_->second.resize(my_images.decoded_pixels_label_it_->second[0]);
		for(uint k = 0; k<my_images.encoded_pixels_label_it_->second.size(); ++k){
			start = my_images.encoded_pixels_label_it_->second[k][my_images.start_pixel_];
			my_images.decoded_pixels_label_it_->second[n_pixel] = start;
			++n_pixel;
			col = std::floor(start/image_data.rows);

			row = start - col*image_data.rows;
			//cout<<start/image_data.rows*image_data.rows<<endl;

			if(row < image_data.rows){
				for(int j = 0; j<my_images.encoded_pixels_label_it_->second[k][my_images.num_pixels_]; ++j){
					if(j+row >= image_data.rows)
						break;
					my_images.decoded_pixels_label_it_->second[n_pixel] = start+j;
					++n_pixel;

					//if(display_mask == true){
					uchar *peg_ptr = image_data.ptr<uchar>(j+row);
					//if(max_j_row < j+row){
					//	max_j_row = j+row;
					//cout<<"max_j_row "<<max_j_row<<endl;
					//}
					//string peg_string(reinterpret_cast<char *>(peg_ptr));
					//cout<<sizeof(peg_ptr)<<" peg_ptr "<<peg_string<<endl;
					peg_ptr[col*n_channels+1] =  (uchar)color;
					/*for(int i = 0; i < image_data.total()*image_data.channels();i+=3){
					if(image_data.data[i+1] == (uchar)color)
						cout<<"i "<< i<<" "<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[i+1]<<" fu "<<float(fu)<<endl;
				}
				cout<<"row*3+col*n_channels "<<row*3+col*n_channels<<" "<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*3+col]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*3+col+1]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*3+col+2]<<endl;

				cout<<"row*col*n_channels "<<row*col*n_channels<<" "<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*col*n_channels]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*col*n_channels+1]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*col*n_channels+2]<<endl;*/

					//}
				}
			}

		}
		n_pixel = 0;
		//break;

		//cv::Mat shaped(224,224,jpegdat.type());
		if(display_mask == true){
					cv::imshow(*labels, image_data);

					//cv::imwrite((image_id+".jpg"),image_data);

					cv::waitKey(0);
					cv::destroyWindow(*labels);
				}

		cv::Mat reshaped(new_shape[0],new_shape[1],jpegdat.type());
		//jpegdat.resize(352);

		cv::resize(image_data,reshaped,reshaped.size());
		//cv::Mat reshaped = jpegdat.reshape(3,224);
		cout<<"before reshape: rows "<<image_data.rows<<" cols "<<image_data.cols<<" size "<<image_data.size<<endl;
		//image_data = image_data.reshape(0,new_shape);
		cout<<"after reshape: rows "<<reshaped.rows<<" cols "<<reshaped.cols<<" size "<<reshaped.size<<endl;


		contents.resize(reshaped.total()*reshaped.channels());

		//uchar *jpg_ptr = jpegdat.data
		for(int k = 0; k < reshaped.total()*reshaped.channels();++k){
			contents[k] = (float)reshaped.data[k];
		}


		//my_images.image_data_vector_it_ = my_images.image_data_vector_.find(image_id);
		my_images.image_data_vector_.insert(std::make_pair(*labels, contents));
		//my_images.image_data_vector_.insert(std::make_pair(image_id, contents));
		//cout<<my_images.image_data_vector_<<endl;
		my_images.image_data_vector_it_ = my_images.image_data_vector_.find(*labels);
		//cout<<my_images.image_data_vector_<<endl;
		cout<<my_images.image_data_vector_it_->second<<endl;

		//my_images.image_data_it_ = my_images.image_data_.find(image_id);
		my_images.image_data_.insert(std::make_pair(*labels, reshaped));

		if(display_mask == true){
			cv::imshow(*labels, reshaped);

			//cv::imwrite((image_id+".jpg"),image_data);

			cv::waitKey(0);
			cv::destroyWindow(*labels);
		}


	}







}
/*
 * Note, instead of using this I can just iterdir function/method for reading zip file and then parse out the char *contents into
 * 		a vector<vector<> >
 *
 */
void Clouds::GetImageData(MyImages1 &my_images,  vector<string> &to_get){


	//my_images.decoded_pixels_.

	cv::Mat blank;
	//this->RLEdecode(blank,my_images);


/*	int n_cores = std::thread::hardware_concurrency();

				//cout<<"modulus "<<std::modulus<double>((double)vect_size/(double)n_cores)<<endl;
				cout<<"modulus "<<vect_size/n_cores<<endl;
				cout<<"modulus "<<vect_size%n_cores<<endl;

				int iter_per_core = vect_size/n_cores;*/

	string model_path = "/home/ryan/resnet50/";//this->file_path_;
	string model_name = "predict_net.pb";//this->model_name;//+".pbtxt";
	string init_model_name = "init_net.pb";//this->init_model_name;//+".pbtxt";



	caffe2::NetDef init_model, predict_model;

	//CAFFE_ENFORCE(caffe2::ReadProtoFromTextFile(model_path+init_model_name, &init_model));
	//CAFFE_ENFORCE(caffe2::ReadProtoFromTextFile(model_path+model_name, &predict_model));

	CAFFE_ENFORCE(caffe2::ReadProtoFromBinaryFile(model_path+init_model_name, &init_model));
	CAFFE_ENFORCE(caffe2::ReadProtoFromBinaryFile(model_path+model_name, &predict_model));

	//cout<<predict_model.DebugString()<<endl;


	string image_id = " ";
	long size  =0;
	long long int a =0;
	char ac;
	std::vector<float> contents;
	std::vector<int> new_shape({224,224});

	for(uint i = 0; i<this->train_imgs_filenames_.size();++i){

		//while(this->train_imgs_filenames_[i].compare("1860780.jpg")!=0)
		//	++i;

		//--i;
		std::vector<std::string> args;
		args.push_back("-p");
		args.push_back(this->train_imgs_path_);
		args.push_back(this->train_imgs_filenames_[i]);


		FILE *infile;

		char buffer [L_tmpnam];
		tmpnam (buffer);
		string tmp_path_and_filename = string(buffer)+".jpg";
		infile = fopen(tmp_path_and_filename.c_str(), "w+b");
		bp::child c("/usr/bin/unzip",args, bp::std_out > infile);

		c.wait();

		fclose(infile);

		cv::Mat jpegdat = cv::imread(tmp_path_and_filename.c_str());

		//cout<<contents[0]<<" "<<jpegdat.at<float>(0,0,0)<<endl;

		image_id = this->train_imgs_filenames_[i].substr(0,this->train_imgs_filenames_[i].find_first_of("."));

		this->ApplyMaskAndRezise(my_images, image_id, to_get,jpegdat, new_shape,false);


		/*//this->train_id_labels_names_.find(imag_id);
		my_images.image_data_vector_.insert(std::make_pair(image_id, contents));
		//cout<<my_images.image_data_vector_<<endl;
		my_images.image_data_vector_it_ = my_images.image_data_vector_.find(image_id);
		//cout<<my_images.image_data_vector_<<endl;
		cout<<my_images.image_data_vector_it_->second<<endl;

		my_images.image_data_.insert(std::make_pair(image_id, jpegdat));


		this->RLEdecode(my_images, image_id, to_get);

		cout<<"before reshape: rows "<<jpegdat.rows<<" cols "<<jpegdat.cols<<" size "<<jpegdat.size<<endl;
		jpegdat = jpegdat.reshape(0,new_shape);
		cout<<"after reshape: rows "<<jpegdat.rows<<" cols "<<jpegdat.cols<<" size "<<jpegdat.size<<endl;


		contents.resize(jpegdat.total()*jpegdat.channels());

		//uchar *jpg_ptr = jpegdat.data
		for(int k = 0; k < jpegdat.total()*jpegdat.channels();++k){
			contents[k] = (float)jpegdat.data[k];
		}


		my_images.image_data_vector_it_ = my_images.image_data_vector_.find(image_id);
		my_images.image_data_vector_.insert(my_images.image_data_vector_it_,std::make_pair(image_id, contents));
		//my_images.image_data_vector_.insert(std::make_pair(image_id, contents));
				//cout<<my_images.image_data_vector_<<endl;
				my_images.image_data_vector_it_ = my_images.image_data_vector_.find(image_id);
				//cout<<my_images.image_data_vector_<<endl;
				cout<<my_images.image_data_vector_it_->second<<endl;

				my_images.image_data_it_ = my_images.image_data_.find(image_id);
				my_images.image_data_.insert(my_images.image_data_it_, std::make_pair(image_id, jpegdat));




		for(int j = 0; j < jpegdat.total()*jpegdat.channels();++j){
			if((uchar)contents[j] != jpegdat.data[j]){
				cout<<j<<" (uchar)contents[j] "<<(float)contents[j]<<" jpegdat.data[j] "<<(float)jpegdat.data[j]<<endl;
			}
		}
*/
		cout<<my_images.decoded_pixels_.begin()->second.size()<<endl;


		remove(tmp_path_and_filename.c_str());

		if(i == 20)
			break;


	}


		//cv::imwrite(tmp_path_and_filename.c_str(),jpegdat);




		cout<<"break"<<endl;


}


/*
 * @brief: performs run length decoding
 * @param: MyImages &my_images object that holds the image data
 * @param: string image_id the id of the image we are decoding
 * @param: bool display_mask to display the mask on the image or not default is false
 *
 */
void Clouds::RLEdecode( MyImages &my_images, string image_id, bool display_mask){

	int start = 0;
	int color = 278;
	int row = 0;
	int col = 0;
	int count = 0;
	int n_pixel = 0;
	vector<int> deocded_tmp;//
	uchar fu = '278';
	//cout<<"fu "<< fu<<" float)fu "<<(float)fu<<endl;
	//cout<<"(float)color "<<(float)color<<endl;

	cv::Mat image_data = my_images.image_data_.find(image_id)->second;
	int n_channels = image_data.channels();
	my_images.encoded_pixels_id_it_ = my_images.encoded_pixels_.find(image_id);
	my_images.decoded_pixels_id_it_ = my_images.decoded_pixels_.find(image_id);

	cout<<my_images.encoded_pixels_id_it_->first<<endl;

	cout<<"shape: cols "<<image_data.cols<<" rows "<<image_data.rows<<" step "<<image_data.step<< endl;

	my_images.encoded_pixels_label_it_  = my_images.encoded_pixels_id_it_->second.begin();
	my_images.decoded_pixels_label_it_  = my_images.decoded_pixels_id_it_->second.begin();
	//add ++my_images.decoded_pixels_label_it_ to the loop below on 11 08 2019
	for(; my_images.encoded_pixels_label_it_!=my_images.encoded_pixels_id_it_->second.end(); ++my_images.encoded_pixels_label_it_ , ++my_images.decoded_pixels_label_it_,++count){


		my_images.decoded_pixels_label_it_->second.resize(my_images.decoded_pixels_label_it_->second[0]);
		for(uint k = 0; k<my_images.encoded_pixels_label_it_->second.size(); ++k){
			start = my_images.encoded_pixels_label_it_->second[k][my_images.start_pixel_];
			my_images.decoded_pixels_label_it_->second[n_pixel] = start;
			++n_pixel;
			col = std::floor(start/image_data.rows);

			row = start - col*image_data.rows;
			//cout<<start/image_data.rows*image_data.rows<<endl;

			for(int j = 0; j<my_images.encoded_pixels_label_it_->second[k][my_images.num_pixels_]; ++j){
				my_images.decoded_pixels_label_it_->second[n_pixel] = start+j;
				++n_pixel;

				if(display_mask == true){
					uchar *peg_ptr = image_data.ptr<uchar>(j+row);
					peg_ptr[col*n_channels+1] =  (uchar)color;
					/*for(int i = 0; i < image_data.total()*image_data.channels();i+=3){
					if(image_data.data[i+1] == (uchar)color)
						cout<<"i "<< i<<" "<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[i+1]<<" fu "<<float(fu)<<endl;
				}
				cout<<"row*3+col*n_channels "<<row*3+col*n_channels<<" "<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*3+col]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*3+col+1]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*3+col+2]<<endl;

				cout<<"row*col*n_channels "<<row*col*n_channels<<" "<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*col*n_channels]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*col*n_channels+1]<<endl;
				cout<<(float)peg_ptr[col*n_channels+1]<<" "<<(float)image_data.data[row*col*n_channels+2]<<endl;*/

				}
			}
		}
		n_pixel = 0;
		break;
	}

	cout<<"after all of shape: cols "<<image_data.cols<<" rows "<<image_data.rows<<" step "<<image_data.step<< endl;

	if(display_mask == true){
		cv::imshow(image_id, image_data);

		//cv::imwrite((image_id+".jpg"),image_data);

		cv::waitKey(0);
		cv::destroyWindow(image_id);
	}



}


/*
 * Note, instead of using this I can just iterdir function/method for reading zip file and then parse out the char *contents into
 * 		a vector<vector<> >
 *
 */
void Clouds::GetImageData(MyImages &my_images){


	//my_images.decoded_pixels_.

	cv::Mat blank;
	//this->RLEdecode(blank,my_images);


/*	int n_cores = std::thread::hardware_concurrency();

				//cout<<"modulus "<<std::modulus<double>((double)vect_size/(double)n_cores)<<endl;
				cout<<"modulus "<<vect_size/n_cores<<endl;
				cout<<"modulus "<<vect_size%n_cores<<endl;

				int iter_per_core = vect_size/n_cores;*/

	string model_path = "/home/ryan/resnet50/";//this->file_path_;
	string model_name = "predict_net.pb";//this->model_name;//+".pbtxt";
	string init_model_name = "init_net.pb";//this->init_model_name;//+".pbtxt";



	caffe2::NetDef init_model, predict_model;

	//CAFFE_ENFORCE(caffe2::ReadProtoFromTextFile(model_path+init_model_name, &init_model));
	//CAFFE_ENFORCE(caffe2::ReadProtoFromTextFile(model_path+model_name, &predict_model));

	CAFFE_ENFORCE(caffe2::ReadProtoFromBinaryFile(model_path+init_model_name, &init_model));
	CAFFE_ENFORCE(caffe2::ReadProtoFromBinaryFile(model_path+model_name, &predict_model));

	//cout<<predict_model.DebugString()<<endl;


	string image_id = " ";
	long size  =0;
	long long int a =0;
	char ac;
	std::vector<float> contents;
	for(uint i = 0; i<this->train_imgs_filenames_.size();++i){

		//while(this->train_imgs_filenames_[i].compare("0011165.jpg")!=0)
		//	++i;

		std::vector<std::string> args;
		args.push_back("-p");
		args.push_back(this->train_imgs_path_);
		args.push_back(this->train_imgs_filenames_[i]);


		FILE *infile;

		char buffer [L_tmpnam];
		tmpnam (buffer);
		string tmp_path_and_filename = string(buffer)+".jpg";
		infile = fopen(tmp_path_and_filename.c_str(), "w+b");
		bp::child c("/usr/bin/unzip",args, bp::std_out > infile);

		c.wait();

		fclose(infile);

		cv::Mat jpegdat = cv::imread(tmp_path_and_filename.c_str());
		cout<<jpegdat.size<<endl;

		contents.resize(jpegdat.total()*jpegdat.channels());

		//uchar *jpg_ptr = jpegdat.data
		for(int i = 0; i < jpegdat.total()*jpegdat.channels();++i){
			contents[i] = (float)jpegdat.data[i];
		}



		cout<<contents[0]<<" "<<jpegdat.at<float>(0,0,0)<<endl;

		image_id = this->train_imgs_filenames_[i].substr(0,this->train_imgs_filenames_[i].find_first_of("."));
		my_images.image_data_vector_.insert(std::make_pair(image_id, contents));
		my_images.image_data_.insert(std::make_pair(image_id, jpegdat));

		this->RLEdecode(my_images, image_id, true);

		for(int i = 0; i < jpegdat.total()*jpegdat.channels();++i){
			if((uchar)contents[i] != jpegdat.data[i]){
				cout<<i<<" (uchar)contents[i] "<<(float)contents[i]<<" jpegdat.data[i] "<<(float)jpegdat.data[i]<<endl;
			}
		}

		cout<<my_images.decoded_pixels_.begin()->second.size()<<endl;


		remove(tmp_path_and_filename.c_str());

		if(i == 10)
			break;


	}


		//cv::imwrite(tmp_path_and_filename.c_str(),jpegdat);




		cout<<"break"<<endl;


}
void Clouds::SetFileNames(string path){

	this->main_path_ = path;
	this->train_imgs_path_= path + "train_images.zip";


}


void Clouds::ParseTrainCSV(string filename, MyImages1 &my_images, int n_get){


	int count = 0;

	int count_get = 0;

	string id;
	string label;
	string id_labels;
	int n_pixels_mask = 0;
	vector<int> tmp_vect2(2);
	vector<vector<int> > tmp_vect;
	vector<vector<string> > fnl_data;
	std::multimap<string, vector<vector<int> > > tmp_map;
	std::multimap<string, vector<int>> tmp_map_decoded;

	typedef const typename std::remove_reference<decltype(my_images.decoded_pixels_.begin()->second)>::type tmp_vect_decoded_type;
	typename std::remove_reference<decltype(my_images.decoded_pixels_.begin()->second)>::type tmp_vect_decoded;

	tmp_vect_decoded.resize(1);

	std::multimap<string, std::multimap<string, tmp_vect_decoded_type > >::iterator  decoded_pixels_itr;
	vector<string > id_and_label(2);
	std::ifstream  data(filename);


	std::multimap<string, std::multimap<string, vector<vector<int> > > >::iterator  encoded_pixels_itr;


	std::string line;

	//this->train_id_labels_names_.resize(train_imgs_filenames_.size());
	//int id_labels_count = 0;


	//to skip header
	std::getline(data,line);
	while(std::getline(data,line))
	{
		std::stringstream  lineStream(line);
		std::string        cell;

		//cout<<"line "<<line<<endl;

		while(std::getline(lineStream,cell,','))
		{
			id = cell.substr(0,cell.find("."));
			label = cell.substr(cell.find("_")+1,cell.size()-cell.find("_"));

			id_labels = id+"_"+label;
			my_images.labels_.insert(label);

			if(cell.size()>=1 && cell.compare("NaN")!=0){

				while(std::getline(lineStream,cell,' ')){
					if(cell.size()>=1 &&count == 0){
						tmp_vect2[my_images.start_pixel_]=stoi(cell);
						++count;
						++n_pixels_mask;
					}
					else if(cell.size()>=1&& count == 1){
						tmp_vect2[my_images.num_pixels_]=stoi(cell);
						tmp_vect.push_back(tmp_vect2);
						n_pixels_mask += stoi(cell);
						count = 0;
					}

				}
				tmp_vect.shrink_to_fit();
				if(tmp_vect.size()>=1 && my_images.encoded_pixels_.find(id)!=my_images.encoded_pixels_.end()){
					//encoded_pixels_itr = my_images.encoded_pixels_.find(id);
					encoded_pixels_itr->second.insert(std::make_pair(id_labels, tmp_vect));

					//vector<int> tmp_vect_decoded(n_pixels_mask);
					tmp_vect_decoded[0] = n_pixels_mask;//was running out of memory by creating the entire vector now so I just put the size here
														//and resize later
					//decoded_pixels_itr = my_images.decoded_pixels_.find(id);
					decoded_pixels_itr->second.insert(std::make_pair(id_labels, tmp_vect_decoded));



				}
				else if(tmp_vect.size()>=1){

					if(tmp_vect.size()>100){
					my_images.encoded_pixels_.insert(std::make_pair(id_labels, tmp_vect));
					//encoded_pixels_itr = my_images1.encoded_pixels_.find(id);
					//my_images.encoded_pixels_.insert(std::make_pair(id, tmp_map));

					//vector<int> tmp_vect_decoded(n_pixels_mask);
					tmp_vect_decoded[0] = n_pixels_mask;//was running out of memory by creating the entire vector now so I just put the size here
														//and resize later
					my_images.decoded_pixels_.insert(std::make_pair(id_labels, tmp_vect_decoded));

					this->id_labels_to_labels.insert(std::make_pair(id_labels,label));
					this->id_labels_to_id.insert(std::make_pair(id_labels,id));
					id_and_label[0] = id;
					id_and_label[1] = label;
					this->id_label_seperated.insert(std::make_pair(id_labels, id_and_label));
					//decoded_pixels_itr = my_images1.decoded_pixels_.find(id);
					//my_images1.decoded_pixels_.insert(std::make_pair(id, tmp_map_decoded));


					//tmp_map.begin()->second.clear();
					//tmp_map.begin()->second.shrink_to_fit();
					//tmp_map.clear();

					//tmp_map_decoded.begin()->second.clear();
					//tmp_map_decoded.begin()->second.shrink_to_fit();
					//tmp_map_decoded.clear();
				}
				}



			}
			else{
				cout<<"Parse line "<<line<<" cell "<<cell<<endl;
				tmp_vect.clear();
				break;
			}
			count++;

		}

		//make sure the labels is not blank
		if(n_pixels_mask>0 &&tmp_vect.size()>100){
			this->train_id_labels_names_it_ =this->train_id_labels_names_.find(id);
			if(this->train_id_labels_names_it_ != this->train_id_labels_names_.end())
				this->train_id_labels_names_it_->second.insert(id_labels);
			else{
				std::set<string> id_labels_name_set;
				id_labels_name_set.insert(id_labels);
				this->train_id_labels_names_.insert(std::make_pair(id, id_labels_name_set));
			}
		}
		n_pixels_mask = 0;
		++count;
		tmp_vect.clear();
		count = 0;
		//if(count_get == n_get)
			//break;
		++count_get;
	}

}

void Clouds::ParseTrainCSV(string filename, MyImages &my_images){


	int count = 0;

	string id;
	string label;
	int n_pixels_mask = 0;
	vector<int> tmp_vect2(2);
	vector<vector<int> > tmp_vect;
	vector<vector<string> > fnl_data;
	std::multimap<string, vector<vector<int> > > tmp_map;
	std::multimap<string, vector<int>> tmp_map_decoded;
	vector<int> tmp_vect_decoded(1);

	std::ifstream  data(filename);


	std::multimap<string, std::multimap<string, vector<vector<int> > > >::iterator  encoded_pixels_itr;
	std::multimap<string, std::multimap<string, vector<int> > >::iterator  decoded_pixels_itr;

	std::string line;


	//to skip header
	std::getline(data,line);
	while(std::getline(data,line))
	{
		std::stringstream  lineStream(line);
		std::string        cell;

		//cout<<"line "<<line<<endl;

		while(std::getline(lineStream,cell,','))
		{
			id = cell.substr(0,cell.find("."));
			label = cell.substr(cell.find("_")+1,cell.size()-cell.find("_"));

			my_images.labels_.insert(label);

			if(cell.size()>=1 && cell.compare("NaN")!=0){

				while(std::getline(lineStream,cell,' ')){
					if(cell.size()>=1 &&count == 0){
						tmp_vect2[my_images.start_pixel_]=stoi(cell);
						++count;
						++n_pixels_mask;
					}
					else if(cell.size()>=1&& count == 1){
						tmp_vect2[my_images.num_pixels_]=stoi(cell);
						tmp_vect.push_back(tmp_vect2);
						n_pixels_mask += stoi(cell);
						count = 0;
					}

				}
				tmp_vect.shrink_to_fit();
				if(tmp_vect.size()>=1 && my_images.encoded_pixels_.find(id)!=my_images.encoded_pixels_.end()){
					encoded_pixels_itr = my_images.encoded_pixels_.find(id);
					encoded_pixels_itr->second.insert(std::make_pair(label, tmp_vect));

					//vector<int> tmp_vect_decoded(n_pixels_mask);
					tmp_vect_decoded[0] = n_pixels_mask;//was running out of memory by creating the entire vector now so I just put the size here
														//and resize later
					decoded_pixels_itr = my_images.decoded_pixels_.find(id);
					decoded_pixels_itr->second.insert(std::make_pair(label, tmp_vect_decoded));
					n_pixels_mask = 0;


				}
				else if(tmp_vect.size()>=1){


					tmp_map.insert(std::make_pair(label, tmp_vect));
					encoded_pixels_itr = my_images.encoded_pixels_.find(id);
					my_images.encoded_pixels_.insert(std::make_pair(id, tmp_map));

					//vector<int> tmp_vect_decoded(n_pixels_mask);
					tmp_vect_decoded[0] = n_pixels_mask;//was running out of memory by creating the entire vector now so I just put the size here
														//and resize later
					tmp_map_decoded.insert(std::make_pair(label, tmp_vect_decoded));
					decoded_pixels_itr = my_images.decoded_pixels_.find(id);
					my_images.decoded_pixels_.insert(std::make_pair(id, tmp_map_decoded));
					n_pixels_mask = 0;

					tmp_map.begin()->second.clear();
					tmp_map.begin()->second.shrink_to_fit();
					tmp_map.clear();

					tmp_map_decoded.begin()->second.clear();
					tmp_map_decoded.begin()->second.shrink_to_fit();
					tmp_map_decoded.clear();
				}



			}
			else{
				cout<<"Parse line "<<line<<" cell "<<cell<<endl;
				tmp_vect.clear();
				break;
			}
			count++;

		}
		//fnl_data.push_back(tmp_vect);
		tmp_vect.clear();
		count = 0;



	}

}

void Clouds::ConvertOnnx2Caffe2(){

	//::onnx_c2
		const char *net = "/home/ryan/resnet18v1/resnet18v1.onnx";
		string onnx_net_loc = "/home/ryan/Downloads/resnet18v1.onnx";



		//::onnx_c2::AttributeProto_AttributeType_AttributeType_MAX


		//::onnx::Checker

		//::ONNX_NAMESPACE::Checker ch;

		//::ONNX_NAMESPACE::Checker

		//::ONNX_NAMESPACE::ModelProto fu;
		//::ONNX_NAMESPACE::


		::onnx_c2::ModelProto onnx_model;
		 std::ifstream in("/home/ryan/resnet18v1/resnet18v1.onnx", std::ios_base::binary);
		 onnx_model.ParseFromIstream(&in);
			        in.close();


			        cout<<"onnx_model.ir_version() "<<onnx_model.ir_version()<<endl;

			        onnx_model.set_ir_version(0x000100000000);
			        cout<<"onnx_model.ir_version() "<<onnx_model.ir_version()<<endl;

		cout<<"Model Version "<<onnx_model.model_version()<<endl;



		//onnx_model.set_ir_version(0x000100010001);
		//cout<<"onnx_model.ir_version() "<<onnx_model.ir_version()<<endl;
		caffe2::onnx::Caffe2Backend back;
		caffe2::NetDef *init_net;
		caffe2::NetDef *pred_net;
		const std::vector<caffe2::onnx::Caffe2Ops> extras;
		 const string my_device_type = caffe2::DeviceTypeName(0);
		 //caffe2::onnx::ConversionContext conversion_cxt;
		 caffe2::onnx::Caffe2BackendRep *my_backend_rep = back.Prepare(onnx_net_loc,my_device_type,extras);
		 cout<<my_backend_rep->init_net().DebugString()<<endl;
		 //back.OnnxToCaffe2(init_net,pred_net, onnx_model, my_device_type,7,true, extras);
		//::onnxifi_library *my_onnx_lib;
		//::onnx_c2::ModelProto model;

		//::onnxifi_load(0,net,my_onnx_lib);

		//std::unique_ptr<ModelProto> model(new ModelProto());
		//onnx::ParseProtoFromBytes(model.get(), bytes, nbytes);
		//::onnxifi_library my_lib;
		//..::onnxifi_load(1,net,my_lib);


	/*	//::ONNX_NAMESPACE::ModelProto model;
	//::onnx_c2::ModelProto model;
		//onnx::ModelProto model;*/
	/*	        std::ifstream in("/home/ryan/resnet18v1/resnet18v1.onnx", std::ios_base::binary);
		        model.ParseFromIstream(&in);
		        in.close();
		        std::cout<<model.graph().input().size()<<" "<<model.graph().DebugString()<<"\n";

		        ::onnxBackend backend;*/
		        //::onnxB




		       // cout<<model.doc_string()<<" "<<model.doc_string()<<endl;
		        //model.DebugString()


}

vector<vector<string> > Clouds::Parse(string file_name, std::set<string> &my_set, int set_column){


	string path = file_name;//"/home/ryan/DI/DITestData/OFER_VDA_BMF_20170814.zip";
	string all_data;

	vector<string> tmp_vect;
	vector<vector<string> > fnl_data;


	//Open the ZIP archive
	int err = 0;
	//std::vector<std::string> filename;
	//filename=iterdir(file_path);

	zip *z = zip_open(path.c_str(), 0, &err);

	//Search for the file of given name
	//string file_name=file_nameTemp;//"OFER_VDA_BMF_20170814.TXT";
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
		//zp = zip_fdopen(ZIP_RDONLY,1,what);

		//Read the compressed file
		//zip_file *f = zip_fopen(z, file_name.c_str(), 0);
		zip_file *f = zip_fopen(z, n, 0);

		if(zip_fread(f, contents, st.size)==-1){
			cout<<"zip_fread did not read anything "<<endl;

//			start_child(contents, st.size, path);

			all_data = contents;
			if(all_data.size()>20)
				cout<<all_data[0]<<endl;
			cout<<"what "<<all_data.size()<<endl;
			if(st.size>100&&all_data.size()<10){
				for( int i=0; i<=st.size;i++)
					all_data.push_back(contents[i]);
			}


		}
		else{
			string fuck = zip_strerror(z);
			//string fuck = zip_file_strerror(f);

			all_data = contents;
			if(all_data.size()>20)
				cout<<all_data[0]<<endl;
			if(st.size>100&&all_data.size()<10){
				for( int i=0; i<=st.size;i++)
					all_data.push_back(contents[i]);
			}

			if(st.size>10)
				zip_fclose(f);

			//And close the archive
			zip_close(z);
		}

		//Do something with the contents
		//delete allocated memory
		delete[] contents;
	}


	return fnl_data;

}

/*
 * @param: string filename of the file to open
 * @param: set<string> my_set a set of the types and or atoms
 * @param: int set_column column number of variable you wan to put into the set
 *
 */
vector<vector<string> > Clouds::Parse(string filename){

	int count = 0;

	vector<string> tmp_vect;
	vector<vector<string> > fnl_data;


	std::ifstream  data(filename);

	std::string line;


	//to skip header
	std::getline(data,line);
	while(std::getline(data,line))
	{
		std::stringstream  lineStream(line);
		std::string        cell;

		//cout<<"line "<<line<<endl;

		while(std::getline(lineStream,cell,','))
		{

			if(cell.size()>=1 && cell.compare("NaN")!=0){

				tmp_vect.push_back(cell);

			}
			else{
				cout<<"Parse line "<<line<<" cell "<<cell<<endl;
				tmp_vect.clear();
				break;
			}
			count++;

		}
		fnl_data.push_back(tmp_vect);
		tmp_vect.clear();
		count = 0;



	}

	return fnl_data;

}



void Clouds::iterdir(string file_path, vector<string> &filename){

	string name;

	int err = 0;
	zip *z = zip_open(file_path.c_str(), 0, &err);

	string zip_error = zip_strerror(z);



	if(z!=NULL){

		struct zip_stat st;
			zip_stat_init(&st);
			//zip_stat(z, file_path.c_str(), 0, &st);

		//char *contents = new char[st.size];

		//int *what=0;
		//struct zip *zp;


		//zp = zip_fdopen(ZIP_RDONLY,1,what);

		//Read the compressed file
		//zip_file *f = zip_fopen(z, file_name.c_str(), 0);
		//zip_file *f = zip_fopen(z, n, 0);

		//if(zip_fread(f, contents, st.size)==-1){
		//	cout<<"did not read zip file"<<endl;

		//}
		//else{

			zip_uint64_t num_file = zip_get_num_entries(z,0);

			zip_error = zip_strerror(z);
			//string image_filename = " ";

			//filename.resize(num_file);
			zip_flags_t zip_flag = 0;

			for(zip_uint64_t i = 0; i<num_file; ++i){
				const char *n = zip_get_name(z,i,zip_flag);
				zip_error = zip_strerror(z);

				//image_filename = n;
				string image_filename(n, std::strlen(n));
				filename.push_back(image_filename);

			}

		//}

			zip_close(z);


	}


}















