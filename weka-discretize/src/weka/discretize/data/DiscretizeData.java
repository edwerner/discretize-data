package weka.discretize.data;

import java.io.BufferedReader;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DiscretizeData {
	Instances data = null;
	NaiveBayes nb;
	
	public static void main(String[] args) {
		DiscretizeData test = new DiscretizeData();
		test.loadFile("data.arff");
		test.generateModel();
		test.saveModel("nb.model");
		test.crossValidate();
	}

	public void loadFile(String arffInput) {
		DataSource source = null;
		try {
			source = new DataSource(arffInput);
			data = source.getDataSet();
			if (data.classIndex() == -1)
				data.setClassIndex(data.numAttributes() - 1);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void generateModel() {
		nb = new NaiveBayes();
		try {
			nb.buildClassifier(data);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void saveModel(String modelPath) {
		try {
			weka.core.SerializationHelper.write(modelPath, nb);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void crossValidate() {
		Evaluation eval = null;
		try {
			eval = new Evaluation(data);
			eval.crossValidateModel(nb, data, 10, new Random(1));
			System.out.println(eval.toSummaryString());
		} catch (Exception e) {
			e.printStackTrace();
		}	
	}
}