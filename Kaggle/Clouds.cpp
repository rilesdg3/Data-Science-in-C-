/*
 * Clouds.cpp
 *
 *  Created on: Oct 7, 2019
 *      Author: ryan
 */

#include <Clouds.h>

Clouds::Clouds() {
	// TODO Auto-generated constructor stub

}

Clouds::~Clouds() {
	// TODO Auto-generated destructor stub
}






void Clouds::CloudsMain(){


	this->iterdir(train_imgs_path_, this->train_imgs_filenames_);

		MyImages my_images;
		this->ParseTrainCSV(this->main_path_+"train.csv", my_images);

		std::multimap<string, std::multimap<string, vector<vector<int> > > >::iterator encoded_pixels_it = my_images.encoded_pixels_.begin();
		std::multimap<string, std::multimap<string, vector<vector<int> > > >::iterator encoded_pixels_end = my_images.encoded_pixels_.end();
		cout<<encoded_pixels_it->first<<endl;


		my_images.encoded_pixels_id_it_ = my_images.encoded_pixels_.find(this->train_imgs_filenames_[0]);
		cout<<my_images.encoded_pixels_.begin()->first<<endl;


				//my_images.encoded_pixels_label_id_ = my_images_pixels_id_it_->second().find()
		if(my_images.encoded_pixels_id_it_ != my_images.encoded_pixels_.end())
				const string fuck =my_images.encoded_pixels_id_it_->first;


		this->GetImageData(my_images);

}

void Clouds::BuildNet(){



//	int n_features = this->features_[0].size();
//		int n_hide = 5;
//		float base_learning_rate = -.0002998;
//		int batch_size =2750;
//		int classes = 1;
//
//		std::cout << "Start training" << std::endl;
//		string model_name = "Test_model";//this->model_name;
//		string init_model_name = "initTest_model;//this->init_model_name;
//
//		string model_path = this->file_path_;
//
//		// >>> model = model_helper.ModelHelper(name="char_rnn")
//		caffe2::NetDef init_model, predict_model;
//		//caffe2::ModelUtil model(init_model, predict_model, model
//		caffe2::ModelUtil model(init_model,predict_model);
//
//		/*
//	flow is
//	1. input
//	2. FC
//	3. activation-> RELU, tanh, sigmoid, softmax
//	4. repeat 2 and 3 for total number of layers
//	5. cost funtion(measure of error rate, or the total loss over all the examples) loss function-> MSE RMSE,
//		 */
//
//
//
//
//		//set activation       model.predict.AddTanhOp("fourth_layer", "tanh3");
//		//std::vector<string > layer({"1"});
//		//vector<int > n_nodes_per_layer({8,16,8,2,3,2,3,5,1,2,4});
//		string layer_name = " ";
//		string activation = "LeakyRelu";//model_config.activation;//"LeakyRelu";//"Tanh";//
//		string layer_in_name = " ";
//		string layer_out_name = " ";
//
//		model.predict.AddInput("input_blob");
//		model.predict.AddInput(activation);
//		model.predict.AddInput("target");
//		//model.predict.AddInput("accuracy");
//		model.predict.AddInput("loss");
//
//		//Add layer, inputs are model to add, name of layer coming in, name of layer going out(i.e. name of this layer??)
//		//number of neurons in this layer,  number of neurons in layer is connection to
//		//think FC does add(matmul(inputs*w,b))
//
//		model.predict.AddStopGradientOp("input_blob");
//
//		int in_size, stride, padding,kernel = 0;
//
//
//
//		for(int i =0; i< model_config.n_hide.size(); ++i){
//			layer_name = std::to_string((i));
//			if(i == 0)
//				model.AddFcOps("input_blob", layer_name, n_features, model_config.n_hide[i]);
//			else
//				model.AddFcOps(activation+std::to_string(i),layer_name,model_config.n_hide[i-1], model_config.n_hide[i]);
//
//			model.AddConvOps(layer_in_name, layer_out_name,in_size, out_size, stride, padding kernel);
//
//			if(activation == "LeakyRelu")
//				model.predict.AddLeakyReluOp(layer_name,activation+std::to_string(i+1),.3);//model.predict.AddSumOp(what, "sum");
//			else if(activation == "Tanh")
//				model.predict.AddTanhOp(layer_name,activation+std::to_string(i+1));
//
//			//cout<<"layer_name "<<layer_name<<" activation+std::to_string(i+1) "<<activation+std::to_string(i+1)<<endl;
//
//		}
//		//layer_name = activation+std::to_string(n_nodes_per_layer.size());
//		layer_name = "last_layer";//"last_layer";//std::to_string(n_nodes_per_layer.size());
//		//cout<<activation+std::to_string(n_nodes_per_layer.size())<<endl;
//		model.AddFcOps(activation+std::to_string(model_config.n_hide.size()),layer_name,model_config.n_hide[model_config.n_hide.size()-1], classes);
//
//		//model.predict.AddConstantFillWithOp(1,"sum","loss");
//		model.init.AddConstantFillOp({1},0.f,"loss");//model.predict.AddConstantFillOp({1},0.f,"loss");
//
//		//had to add this so I could usev train.AddSgdOps();
//		model.init.AddConstantFillOp({1},0.f,"one");//model.predict.AddConstantFillOp({1},0.f,"loss");
//
//		//model.init.AddConstantFillWithOp(1.f, "loss", "loss_grad");
//		//set loss
//		//model.predict.AddSquaredL2Op(layer_name,"target","sql2");
//		model.predict.AddL1DistanceOp(layer_name,"target","sql2");
//
//		//model.predict.net.A
//		model.predict.AddAveragedLossOp("sql2", "loss");
//
//
//		model.AddIterOps();
//
//		caffe2::NetDef f_int = model.init.net;
//		caffe2::NetDef pred = model.predict.net;
//		caffe2::ModelUtil save_model(f_int, pred, model_name);
//
//
//		//cout<<model.predict.net.DebugString()<<endl;
//		/*	caffe2::NetDef train_model(model.predict.net);
//		caffe2::NetUtil train(train_model, "train");*/
//		caffe2::NetDef train_init, train_predict;
//		caffe2::ModelUtil train(train_init, train_predict,"train");
//		string su = "relu";
//		model.CopyTrain(layer_name, 1,train);
//
//		//train.predict.AddInput("iter");//this should some how get done in void ModelUtil::CopyTrain(const std::string &layer, int out_size,ModelUtil &train)
//		//train.predict.AddInput("one");//this should some how get done in void ModelUtil::CopyTrain(const std::string &layer, int out_size,ModelUtil &train)
//		train.predict.AddConstantFillWithOp(1.f, "loss", "loss_grad");
//
//		//set optimizer
//		//model.AddAdamOps();
//		//model.AddRmsPropOps();
//		train.predict.AddGradientOps();
//		base_learning_rate = -1*base_learning_rate;
//		train.predict.AddLearningRateOp("iter","lr",base_learning_rate,.9);
//		train.AddSgdOps();
//		//train.AddRmsPropOps();
//
//
//		/*cout<<model.init.Proto()<<endl;
//		cout<<endl;
//		cout<<model.predict.Proto()<<endl;
//		cout<<endl;
//		cout<<train.init.Proto()<<endl;
//		cout<<endl;
//		cout<<train.predict.Proto()<<endl;*/
//		//Start training
//		caffe2::Workspace workspace("tmp");
//
//		// >>> log.debug("Training model")
//		std::cout << "Train model" << std::endl;
//
//		// >>> workspace.RunNetOnce(self.model.param_init_net)
//		CAFFE_ENFORCE(workspace.RunNetOnce(model.init.net));
//
//		auto epoch = 0;
//
//		// >>> CreateNetOnce(self.model.net)
//		workspace.CreateBlob("input_blob");
//		//workspace.CreateBlob("accuracy");
//		workspace.CreateBlob("loss");
//		workspace.CreateBlob("target");
//		//workspace.CreateBlob("one");
//
//		workspace.CreateBlob("lr");
//
//		CAFFE_ENFORCE(workspace.RunNetOnce(train.init.net));
//		CAFFE_ENFORCE(workspace.CreateNet(train.predict.net));//CAFFE_ENFORCE(workspace.CreateNet(train.net));
//
//		//cout<<train.init.net.name()<<" "<<model.init.net.name()<<endl;
//
//		CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));//CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));
//
//		float wFbefore = caffe2::BlobUtil(*workspace.GetBlob("2_w")).Get().data<float>()[0];
//		float wFafter = 0.00000000;
//		//	float wSefore = caffe2::BlobUtil(*workspace.GetBlob("1_w")).Get().data<float>()[0];
//		//float wLefore = BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0];
//
//		int nTrainBatches = batch_size;//FLAGS_batch;//TrainData.Features.size()/FLAGS_batch;
//
//		//CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));
//		std::vector<float> tmp;
//
//		//compute number of minibatches for training, validation and testing
//		int n_train_batches =data.train_features_.size()/ batch_size;//TrainData.Features.size() / batch_size;
//		int n_valid_batches = data.validate_features_.size()/ batch_size;// ValidateData.Features.size() / batch_size;
//		int n_test_batches = data.test_features_.size()/ batch_size;//TestData.Features.size() / batch_size;
//
//		//early-stopping parameters
//		int patience = 4000;//  # look as this many examples regardless
//		int patience_increase = 4;//  # wait this much longer when a new best is found
//		float improvement_threshold = 0.595; //a relative improvement of this much is
//		//considered significant
//		int validation_frequency = std::min(n_train_batches, patience / 2);
//
//		int iter= 0;
//
//		bool done_looping = false;
//		int server = 0;
//
//		std::vector<float> validation_losses;
//		float this_validation_loss =0.0;
//		float best_validation_loss = std::numeric_limits<float>::max();
//		int best_iter =0;
//		float test_score =0.0;
//
//		float train_score = 0.0;
//
//		vector<float>tmp_w(n_features);
//		vector<vector<float> > check_weights_w;
//
//		while((best_test_score > model_config.no_train_thresh)){
//			CAFFE_ENFORCE(workspace.RunNetOnce(train.init.net));
//			done_looping = false;
//			iter = 0;
//			patience = 4000;
//
//			std::vector<float> validation_losses;
//			this_validation_loss =0.0;
//			best_validation_loss = std::numeric_limits<float>::max();
//			best_iter =0;
//			test_score =0.0;
//
//			train_score = 0.0;
//			epoch = 0;
//			//while (epoch < n_epochs) and (not done_looping):
//			while (epoch < FLAGS_iters && !done_looping) {
//
//				epoch++;//this for like total number of iterations
//
//				//Train
//				for(auto minibatch_index = 0; minibatch_index<(n_train_batches); ++minibatch_index ){
//
//
//					{
//						std::vector<int> dim({nTrainBatches,data.train_features_[minibatch_index].size()});
//						caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, data.train_features_, minibatch_index, false);
//					}
//
//					{
//						std::vector<int> dim({nTrainBatches,1});
//						caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, data.train_labels_, minibatch_index, false);
//
//						//std::cout<<"Train Label "<<caffe2::BlobUtil(*workspace.GetBlob("target")).Get().DebugString()<<std::endl;
//
//					}
//
//
//					CAFFE_ENFORCE(workspace.RunNet(train.predict.net.name()));//CAFFE_ENFORCE(workspace.RunNet(train.net.name()));
//
//					/*			cout<<caffe2::BlobUtil(*workspace.GetBlob(layer_name)).Get().DebugString()<<endl;//data<float>()[0];
//				for(int i = 0; i<nTrainBatches; ++i)
//					cout<<"layr_name data "<<caffe2::BlobUtil(*workspace.GetBlob(layer_name)).Get().data<float>()[i]<<endl;
//
//				for(int i =0; i<n_features; ++i)
//						tmp_w[i] = caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[i];//cout<<caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[i]<<" ";
//					check_weights_w.push_back(tmp_w);
//					cout<<endl;*/
//					//cout<<"wFbefore "<<wFbefore <<" After "<< caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[0]<<endl;
//					//			cout<<"wSefore "<<wSefore<<" After "<< caffe2::BlobUtil(*workspace.GetBlob("1_w")).Get().data<float>()[0]<<endl;
//					//	cout<<"wLefore "<<wLefore<<" After "<< BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0]<<endl;
//
//
//
//
//					train_score = caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];// cout<<"Train Score "<<train_score<<endl;
//
//					iter = (epoch - 1) * n_train_batches + minibatch_index;
//
//					if((iter + 1) % validation_frequency == 0){
//
//						{
//							//std::vector<int> dim({nTrainBatches,ValidateData.Features[minibatch_index].size()});
//							std::vector<int> dim({nTrainBatches,data.validate_features_[minibatch_index].size()});
//							//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, ValidateData.Features, minibatch_index,false);
//							caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, data.validate_features_, minibatch_index,false);
//						}
//
//						{
//							std::vector<int> dim({nTrainBatches,1});
//							//BlobUtil(*workspace.CreateBlob("target")).Set(dim, ValidateData.Labels,minibatch_index, false);
//							caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, data.validate_labels_,minibatch_index, false);
//							//std::cout<<"Validate Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
//						}
//
//
//						// >>> workspace.RunNet(self.forward_net.Name())
//						CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));
//
//						//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;
//
//
//						//cout<<"Validate accuracy "<<1.0-BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0]<<endl;
//
//						//this_validation_loss =1.0-caffe2::BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
//						this_validation_loss =caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
//						//cout<<"Valied Score "<<this_validation_loss<<endl;// 1-(float)(nCorrect)/(float)(countValue);//sess.run([accuracy], feed_dict={X: batch_x, Y: batch_y})[0]#numpy.mean(validation_losses)
//						//cout<<"Validate Percent Correct "<<(float)(nCorrect)/(float)(countValue)<<endl;
//
//						validation_losses.push_back(this_validation_loss);
//						// if we got the best validation score until now
//						if(this_validation_loss < best_validation_loss){
//							//improve patience if loss improvement is good enough
//							if( this_validation_loss < best_validation_loss * improvement_threshold)
//								patience = max(patience, iter * patience_increase);
//
//							best_validation_loss = this_validation_loss;
//							best_iter = iter;
//
//							//cout<<"optimizer_iteration "<<BlobUtil(*workspace.GetBlob("optimizer_iteration")).Get() <<endl;
//
//							{
//								//std::vector<int> dim({nTrainBatches,TestData.Features[minibatch_index].size()});
//								std::vector<int> dim({nTrainBatches,data.test_features_[minibatch_index].size()});
//								//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, TestData.Features,minibatch_index, false);
//								caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, data.test_features_,minibatch_index, false);
//							}
//
//							{
//								std::vector<int> dim({nTrainBatches,1});
//								//BlobUtil(*workspace.CreateBlob("target")).Set(dim, TestData.Labels, minibatch_index, false);
//								caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, data.test_labels_, minibatch_index, false);
//								//std::cout<<"Test Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
//							}
//
//							// >>> workspace.RunNet(self.forward_net.Name())
//							CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));
//
//							//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;
//
//							test_score = caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
//							//cout<<"Test Score "<<test_score<<endl;//(float)(nCorrect)/(float)(countValue);
//
//							cout<<"Train "<<train_score<< " Validate "<<this_validation_loss<<" Test "<<test_score<<endl;
//
//							//cout<<"wFbefore "<<wFbefore*10000 <<" After "<< caffe2::BlobUtil(*workspace.GetBlob("2_w")).Get().data<float>()[0]*10000<<endl;
//							//cout<<"Test accuracy "<<1.0-BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0]<<endl;
//
//							caffe2::NetDef deploy_init_model;  // the final initialization model
//							caffe2::ModelUtil deploy(deploy_init_model, save_model.predict.net,model.init.net.name());
//							//caffe2::ModelUtil deploy(deploy_init_model, model.predict.net,model.init.net.name());
//
//
//							save_model.CopyDeploy(deploy, workspace);
//							//CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));
//							if(server == 0){
//								//cout<<"Save Model"<<endl;//mdir=saver.save(sess, save_path='/home/ryan/workspace/MultiLayerTen/'+modelName, global_step=tf.train.global_step(sess,global_step_tensor))
//								caffe2::WriteProtoToBinaryFile(deploy.init.net, model_path+init_model_name);//caffe2::WriteProtoToTextFile(deploy.init.net, model_path+init_model_name+".pbtxt");//
//								caffe2::WriteProtoToBinaryFile(model.predict.net, model_path+ model_name);//caffe2::WriteProtoToTextFile(model.predict.net, model_path+ model_name+".pbtxt");//
//
//							}
//							else{
//								//cout<<"Save Model"<<endl;//mdir=saver.save(sess, save_path='/home/riley/data/HE/HE1-2Models/'+modelName, global_step=tf.train.global_step(sess,global_step_tensor))
//								caffe2::WriteProtoToTextFile(model.init.net, "/home/ryan/workspace/adsf/initModel");
//								caffe2::WriteProtoToTextFile(model.predict.net, "/home/ryan/workspace/adsf/model1");
//							}
//							//save the best model
//
//						}
//
//					}
//
//
//					if(patience <= iter){
//						done_looping = true;
//						break;
//					}
//
//				}
//
//
//				//std::cout << "Smooth loss: " << smooth_loss << std::endl;
//				//std::cout<<"last_layer_w After Training "<<BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0]<<std::endl;
//				if((test_score+this_validation_loss)/2<this->best_test_score){
//					best_test_score = (test_score+this_validation_loss)/2;
//					cout<<"Best Test Score "<<best_test_score<<endl;
//				}
//
//			}
//		}
//



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

void Clouds::RLEdecode( MyImages &my_images, string image_id){

	int start = 0;
	int color = 138;

	cv::Mat image_data = my_images.image_data_.find(image_id)->second;
	int n_channels = image_data.channels();
	my_images.encoded_pixels_id_it_ = my_images.encoded_pixels_.find(image_id);

	cout<<my_images.encoded_pixels_id_it_->first<<endl;

	int row = 0;
	int col = 0;
	my_images.encoded_pixels_label_id_  = my_images.encoded_pixels_id_it_->second.begin();
	for(; my_images.encoded_pixels_label_id_!=my_images.encoded_pixels_id_it_->second.end(); ++my_images.encoded_pixels_label_id_){


		for(uint k = 0; k<my_images.encoded_pixels_label_id_->second.size(); ++k){
			start = my_images.encoded_pixels_label_id_->second[k][my_images.start_pixel_];


			col = std::floor(start/image_data.rows);//std::floor(start/jpegdat.cols)+row;

			row = start - col*image_data.rows;

			for(int j = 0; j<my_images.encoded_pixels_label_id_->second[k][my_images.num_pixels_]; ++j){

				uchar *peg_ptr = image_data.ptr<uchar>(j+row);
				peg_ptr[col*n_channels+1] =  (uchar)color;

			}
		}
	}

	cout<<"after all of shape: cols "<<image_data.cols<<" rows "<<image_data.rows<<" step "<<image_data.step<< endl;

	cv::imshow("image", image_data);

	cv::waitKey(0);
	cv::destroyWindow("image");


}



void Clouds::AhShit(cv::Mat &image, MyImages &my_images){

	int start = 0;
		int zero = 0;
		int color = 138;
		for(uint i = 0; i<this->train_imgs_filenames_.size();++i){

			while(this->train_imgs_filenames_[i].compare("0011165.jpg")!=0)
				++i;

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


			my_images.encoded_pixels_id_it_ = my_images.encoded_pixels_.find(this->train_imgs_filenames_[i].substr(0,this->train_imgs_filenames_[i].find_first_of(".")));
			//my_images.encoded_pixels_id_it_ = my_images.encoded_pixels_.find(my_images.encoded_pixels_.begin()->first);
			cout<<my_images.encoded_pixels_id_it_->first<<endl;


			cout<<"jpegdat.type() "<<cv::typeToString(jpegdat.type())<<endl;

			//cout<<"size.dims() "<<jpegdat.size.dims()<<endl;
			//jpegdat.rowRange()
			//cout<<" jpegdat.size "<<jpegdat.size<<endl;
			//jpegdat.resize(150528);


			cv::Size size(jpegdat.rows*jpegdat.cols,1);
			std::vector<int> newshape(2);
			newshape[0] = 1;
			newshape[1] = (jpegdat.cols*jpegdat.channels())*jpegdat.rows;






			cv::Mat fuckmat;
			fuckmat.create(size,0);
					fuckmat.zeros(0,jpegdat.rows*jpegdat.cols, 0);

					cout<<"fuck.type() "<<cv::typeToString(fuckmat.type())<<endl;
					cout<<"is continuous "<<fuckmat.isContinuous()<<endl;

			//		fuckmat.reshape(jpegdat.rows,jpegdat.cols,0);
			//fuckmat.zeros(jpegdat.rows,jpegdat.cols, jpegdat.type());
			//unsigned char *input = (unsigned char*)(fuckmat.data);

			cout<<fuckmat.step<<endl;
			vector<cv::Point> fuck_point;
			//jpegdat = jpegdat.reshape(1,newshape);
			cout<<"jpegdat shape start: rows "<<jpegdat.rows<< " columns "<<jpegdat.cols<<" channels "<<jpegdat.channels()<<" step "<<jpegdat.step<<" depth "<<jpegdat.depth()<<endl;
			cout<<"fuckmat shape start: rows "<<fuckmat.rows<< " columns "<<fuckmat.cols<<" channels "<<fuckmat.channels()<<" step "<<fuckmat.step<<" depth "<<fuckmat.depth()<<endl;
			//unsigned char *input = (unsigned char*)(fuckmat.data);
			//unsigned char *input = (unsigned char*)(fuckmat.data);

			vector<int> col_vect(jpegdat.cols);
			vector<vector<int> > row_vect(jpegdat.rows, col_vect);
			int count = 0;
			my_images.encoded_pixels_label_id_  = my_images.encoded_pixels_id_it_->second.begin();
			for(; my_images.encoded_pixels_label_id_!=my_images.encoded_pixels_id_it_->second.end(); ++my_images.encoded_pixels_label_id_){
				cout<<my_images.encoded_pixels_label_id_->first<<endl;
				cout<<(fuckmat.cols*fuckmat.channels())*fuckmat.rows<<endl;
				cout<<fuckmat.total()<<endl;
				//for(int q=0; q!=(jpegdat.cols*jpegdat.channels())*jpegdat.rows; ++q){
				//	input[q] = color;
				//}

				//break;
				//for(uint i = 0; i<my_images.encoded_pixels_label_id_->second.size(); ++i){
				//cout<<"my "<<my_images.encoded_pixels_id_it_->second.size()<<endl;
				//cout<<my_images.encoded_pixels_id_it_->second.begin()->second[i][my_images.num_pixels_]<<endl;
				//cout<<"rows "<<jpegdat.rows<< " columns "<<jpegdat.cols<<" channels "<<jpegdat.channels()<<endl;



				cout<<"is continuous "<<fuckmat.isContinuous()<<endl;
				for(uint k = 0; k<my_images.encoded_pixels_label_id_->second.size(); ++k){
					start = my_images.encoded_pixels_label_id_->second[k][my_images.start_pixel_];

					//for(int q=0; q<=937*3; ++q)
					//	input[start*3+q*3] = (uchar)255;
					//break;
					//fuckmat[cv::Range(start,my_images.encoded_pixels_label_id_->second[k][my_images.start_pixel_])];
					//cout<<my_images.encoded_pixels_id_it_->second.[k][my_images.num_pixels_]<<endl;
					/*for (int i = 0; i < jpegdat.rows; i+=20)
				    cv::line(jpegdat, cv::Point(0,i), cv::Point(jpegdat.cols-1, i), cv::Scalar(255,0,0), 1);

				for (int j = 0; j < jpegdat.cols; j+=20)
					cv::line(jpegdat, cv::Point(j,0),cv::Point(j, jpegdat.rows-1), cv::Scalar(255,0,0), 1);*/


					if(k>=my_images.encoded_pixels_label_id_->second.size())
						cout<<"waht"<<endl;
					//cout<<my_images.encoded_pixels_label_id_->second[k][my_images.num_pixels_]<<endl;
					//for(int row  = 0; row< jpegdat.rows; ++row){
					//	for(int col =0; col<jpegdat.cols; ++col){
					//		if(row*col == start){
								//{//input[start+1] =(uchar)zero;{
								for(int j = 0; j<my_images.encoded_pixels_label_id_->second[k][my_images.num_pixels_]; ++j){

									//if(j == 0)
									//	cout<<"start "<<start<<" j "<<j<<endl;
									//cout<<"start+j "<<start+j<<endl;
									//jpegdat.at<uchar>(start*3+j)=(uchar)zero;
	//								input[start+j ] =(uchar)color;
									//input[jpegdat.step * row + col*jpegdat.channels()+1 ] =(uchar)zero;
									//fucker[0] = (uchar)zero;
									//input[(start+j)*3+1] = (uchar)zero;
									//input[6300+start*3+j*3] = (uchar)color;
									//input[start+j+1]= (uchar)color;
									//input[start+j+2]= (uchar)zero;
									//jpegdat.at<int>(start,start+j+1) = 255;//cout<<"fuck"<<endl;
								}
							//}
						//}

					//}

								//cout<<"start "<<start<<" (int)input[start] "<<(int)input[start]<<endl;
				}

				break;
			}


			std::vector<int> final_shape(2);
					final_shape[0] = 1400;//rows
					final_shape[1] = 2100;//colums

			//fuckmat.re
			cv::Mat img(100, 100, CV_8UC3);
			cv::Mat padded;
			int padding = 3;
			padded.create(fuckmat.rows + 2*padding, fuckmat.cols + 2*padding, fuckmat.type());
			padded.setTo(cv::Scalar::all(0));

			//fuckmat.copyTo(padded(cv::Rect(padding, padding, fuckmat.cols, fuckmat.rows)));

			cout<<"padded shape: rows "<<padded.rows<<" cols "<<padded.cols<<" channels "<<padded.channels()<<" step "<<padded.step<<" depth "<<padded.depth()<<endl;

			//padded= padded.reshape(3,final_shape);
			//cout<<"padded reshape: rows "<<padded.rows<<" cols "<<padded.cols<<" channels "<<padded.channels()<<" step "<<padded.step<<" depth "<<padded.depth()<<endl;


			//jpegdat = jpegdat.reshape(3,final_shape);
			fuckmat = fuckmat.reshape(3,final_shape);
			cout<<fuckmat.at<int>(317,189)<<endl;;

			//cv::Mat source = cv::imread(path);

			//cv::Mat newSrc(jpegdat.size(), CV_MAKE_TYPE(fuckmat.depth(), 3));

			int from_to[] = { 0,0};//, 1,1, 2,2};


			//cv::mixChannels(&fuckmat,1,&newSrc,1,from_to,3);

			//cv::mixChannels(&fuckmat,1,&jpegdat,1,from_to,1);

			cv::Mat in_mat;   // Already created
			cv::Mat mask_mat; // Already created
			cv::Mat out_mat;  // New and empty

			unsigned char *input = (unsigned char*)(jpegdat.data);

			//for(int q=0; q!=(jpegdat.cols*jpegdat.channels())*jpegdat.rows; ++q){
			//				input[q] = color;
			//			}

			//cv::Vec3b my_vec3b = jpegdat.at<cv::Vec3b>(317,189);
			int zero = 0;
			uchar *peg_ptr = jpegdat.ptr<uchar>(317);

			peg_ptr[100] =  (uchar)color;
			peg_ptr[100*3+1] =  (uchar)color;
			peg_ptr[189*3+1] =  (uchar)color;

			//cout<<"newSrc shape: rows "<<newSrc.rows<<" cols "<<newSrc.cols<<" channels "<<newSrc.channels()<<" step "<<newSrc.step<<" depth "<<newSrc.depth()<<endl;

			cout<<"after all of shape: rows "<<fuckmat.rows<<" cols "<<fuckmat.cols<<" channels "<<fuckmat.channels()<<" step "<<fuckmat.step<<" depth "<<fuckmat.depth()<<endl;

			//fuckmat.copyTo(out_mat, jpegdat);

			cv::imshow("image", jpegdat);

			cv::waitKey(0);
			cv::destroyWindow("image");

			remove(tmp_path_and_filename.c_str());

			if(i == 1000)
				break;


		}


}
/*
 * Note, instead of using this I can just iterdir function/method for reading zip file and then parse out the char *contents into
 * 		a vector<vector<> >
 *
 */
void Clouds::GetImageData(MyImages &my_images){

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

		image_id = this->train_imgs_filenames_[i].substr(0,this->train_imgs_filenames_[i].find_first_of("."));

		my_images.image_data_.insert(std::make_pair(image_id, jpegdat));

		this->RLEdecode(my_images, image_id);



		remove(tmp_path_and_filename.c_str());

		if(i == 1000)
			break;


	}


		//cv::imwrite(tmp_path_and_filename.c_str(),jpegdat);




		cout<<"break"<<endl;


}
void Clouds::SetFileNames(string path){

	this->main_path_ = path;
	this->train_imgs_path_= path + "train_images.zip";


}

void Clouds::ParseTrainCSV(string filename, MyImages &my_images){


	int count = 0;

	string id;
	string label;
	vector<int> tmp_vect2(2);
	vector<vector<int> > tmp_vect;
	vector<vector<string> > fnl_data;
	std::multimap<string, vector<vector<int> > > tmp_map;

	std::ifstream  data(filename);


	std::multimap<string, std::multimap<string, vector<vector<int> > > >::iterator  encoded_pixels_itr;

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

			if(cell.size()>=1 && cell.compare("NaN")!=0){

				while(std::getline(lineStream,cell,' ')){
					if(cell.size()>=1 &&count == 0){
						tmp_vect2[my_images.start_pixel_]=stoi(cell);
						++count;
					}
					else if(cell.size()>=1&& count == 1){
						tmp_vect2[my_images.num_pixels_]=stoi(cell);
						tmp_vect.push_back(tmp_vect2);
						count = 0;
					}

				}
				tmp_vect.shrink_to_fit();
				if(tmp_vect.size()>=1 && my_images.encoded_pixels_.find(id)!=my_images.encoded_pixels_.end()){
					encoded_pixels_itr = my_images.encoded_pixels_.find(id);
					encoded_pixels_itr->second.insert(std::make_pair(label, tmp_vect));

				}
				else if(tmp_vect.size()>=1){


					tmp_map.insert(std::make_pair(label, tmp_vect));
					encoded_pixels_itr = my_images.encoded_pixels_.find(id);
					my_images.encoded_pixels_.insert(std::make_pair(id, tmp_map));

					tmp_map.begin()->second.clear();
										tmp_map.begin()->second.shrink_to_fit();
										tmp_map.clear();
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















