package weka.discretize.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class DiscretizeData {
	static Instances data = null;
	NaiveBayes nb;

	public static void main(String[] args) {
		DiscretizeData test = new DiscretizeData();
		test.loadFile("data.arff");
		test.generateModel();
		test.saveModel("nb.model");
		test.crossValidate();
		try {
			discretizeData();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void discretizeData() throws Exception {
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("data.csv"));
		Instances instances = loader.getDataSet();
		// Make the last attribute be the class
		instances.setClassIndex(instances.numAttributes() - 1);

		/*
		 * Replace missing values
		 */
		ReplaceMissingValues fixMissing = new ReplaceMissingValues();
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);

		/*
		 * Discretize data
		 */
		Discretize discretizeNumeric = new Discretize();
		discretizeNumeric.setOptions(new String[] { "-R", "first-last" });
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);

		// Make the last attribute be the class
		instances.setClassIndex(instances.numAttributes() - 1);
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