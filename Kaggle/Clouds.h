/*
 * Clouds.h
 *
 *  Created on: Oct 7, 2019
 *      Author: ryan
 */

#ifndef CLOUDS_CLOUDS_H_
#define CLOUDS_CLOUDS_H_

#include "Data.h"
#include "Stats.h"
#include "Ensemble.h"

#include <caffe2/core/init.h>
#include <caffe2/core/tensor.h>

#include "blob.h"
#include "model.h"
#include "net.h"


#include <limits>
#include <numeric>
#include <sys/stat.h>

#include "boost/process.hpp"

#include <jpeglib.h>
#include <jerror.h>
#include <setjmp.h>

#include "grfmt_jpeg.hpp"
#include "utils.hpp"


// the below will work but I get some ir version error
//#include <caffe2/proto/caffe2_pb.h> //I used this
#include <onnx/onnx_pb.h>
#include <onnx/onnx-operators_pb.h>
#include <onnx/onnx_onnx_c2.pb.h>
//#include <caffe2/onnx/backend.h>

//#include <onnx/onnx-operators.proto3.pb.h>
//#include <onnx/onnx.proto3.pb.h>
//#include <onnx/onnxifi.h>
//#include <onnx/onnxifi_loader.h>
//#include <onnx/onnxifi_utils.h>

//#include <onnx/proto_utils.h>
//#include <onnx/onnx-operators_pb.h>
//#include <onnx/onnx_onnx_c2.pb.h>
//#include <caffe2/onnx/backend.h>

//#include <onnx/checker.h>
//#include <onnx/string_utils.h>
//#include <onnx/common/status.h>
//#include <onnx/onnx-ml.pb.h>
//#include <onnx/defs/function.h>
//#include <python3.6/dist-packages/onnx/defs/attr_proto_util.h>

//#include <onnx/defs/function.h>///defs/function.h:17:57: error: ‘FunctionProto’ was not declared in this scope
//#include <onnx/defs/data_type_utils.h>///data_type_utils.h:39:32: error: ‘TypeProto’ does not name a type

//#include <onnx/shape_inference/implementation.h>

//#include <onnx/common/constants.h>
//#include <onnx/common/assertions.h>
//#include "onnx/version_converter/adapters/adapter.h"
//#include <onnx/common/tensor.h>
//#include <onnx/checker.h>//defs/function.h:17:57: error: ‘FunctionProto’ was not declared in this scope
//#include <onnx/version_converter/adapters/adapter.h>



//#include <onnx/common/constants.h>
//#include <onnx/version_converter/helper.h>
//#include <onnx/version_converter/convert.h>
//#include <onnx/common/ir.h>
//#include <onnx/defs/shape_inference.h>
//#include <onnx/defs/schema.h> //error: ‘TypeProto’ does not name a type


#include <caffe2/onnx/backend.h>//caffe2/onnx/helper.h:13:25: error: ‘ONNX_NAMESPACE::AttributeProto’ has not been declared

//#include "onnx/onnxifi_utils.h"
//#include <onnx/onnx-operators_onnx_c2.pb.h>

//#include <onnx/version_converter/BaseConverter.h>

//#include <onnx/version_converter/helper.h>

//#include "onnx/onnx-operators.pb.h"


//#include <onnx/version_converter/convert.h>


//#include <onnx/onnx.pb.h>

//#include <onnx/proto_utils.h>

//#include <onnx/onnxifi_utils.h> //proto_utils.h:38:65: error: ‘AttributeProto’ does not name a type

//#include <third_party/onnx/onnxbackend.h>



#include <opencv2/imgproc.hpp>



namespace bp = ::boost::process;



struct MyImages{


	/*
		 * string image id
		 * cv::Mat data of image
		 */
		std::multimap<string, cv::Mat> image_data_;
		/*
		 * string image id
		 * string image label
		 * vector<vector< int > > incoded pixels start_pixel, num_pixels
		 */
		std::multimap<string, std::multimap<string, vector<vector<int> > > >  encoded_pixels_;
		std::multimap<string, std::multimap<string, vector<vector<int> > > >::iterator  encoded_pixels_id_it_;
		std::multimap<string, vector<vector<int> > >::iterator  encoded_pixels_label_id_;
		const int start_pixel_=0;//
		const int num_pixels_ =1;


};


class Clouds {


public:
	Clouds();
	virtual ~Clouds();


	string main_path_;
	string train_imgs_path_;
	vector<string> train_imgs_filenames_;
	string train_labels_filename_;
	vector<string> test_imgs_filenames_;

	std::set<string> labels_{"Fish", "Flower", "Gravel", "Sugar"};

	void CloudsMain();
	void BuildNet();
	void ConvertOnnx2Caffe2();
	void RLEdecode(MyImages &my_images, string image_id);
	void AhShit(cv::Mat &image, MyImages &my_images);
	void GetImageData(MyImages &my_images);
	void SetFileNames(string path);
	void ParseTrainCSV(string filename, MyImages &my_images);
	void iterdir(string file_path, vector<string> &filename);
	vector<vector<string> > Parse(string file_name, std::set<string> &my_set, int set_column);
	vector<vector<string> > Parse(string filename);


};





class ImageInfo{



};


















#endif /* CLOUDS_CLOUDS_H_ */
