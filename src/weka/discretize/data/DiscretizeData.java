package weka.discretize.data;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * The Class DiscretizeData.
 */
public class DiscretizeData {
	
	private static Instances data;
	private NaiveBayes naiveBayes;

	/**
	 * Instantiate DiscretizeData class, 
	 * discretize data loaded from file, 
	 * generate and save naive Bayes model
	 * and validate model.
	 *
	 * @param args the arguments
	 * @throws Exception the exception
	 */
	public static void main(String[] args) throws Exception {
		DiscretizeData discretize = new DiscretizeData();
		discretize.loadDataFile("data.arff");
		discretize.discretizeData();
		discretize.generateModel();
		discretize.saveModel("naiveBayes.model");
		discretize.crossValidate();
	}

	/**
	 * Load data file.
	 *
	 * @param input the input
	 */
	public void loadDataFile(String input) {
		DataSource source = null;
		try {
			source = new DataSource(input);
			data = source.getDataSet();
			if (data.classIndex() == -1) {
				data.setClassIndex(data.numAttributes() - 1);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Discretize data loaded from CSV file, 
	 * set class index for last attribute in
	 * file, replace missing values and 
	 * discretize data and set options and import
	 * format for data.
	 *
	 * @return the instances
	 * @throws Exception the exception
	 */
	public void discretizeData() throws Exception {

		// remove useless values
		RemoveUseless remove = new RemoveUseless();
		remove.setOptions(new String[] { "-M", "99" });
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);

		// replace missing values
		ReplaceMissingValues replace = new ReplaceMissingValues();
		replace.setInputFormat(data);
		data = Filter.useFilter(data, replace);

		// Discretize data
		Discretize discretize = new Discretize();
		discretize.setOptions(new String[] { "-R", "first-last" });
		discretize.setInputFormat(data);
		data = Filter.useFilter(data, discretize);
	}

	/**
	 * Generate and build naive Bayes
	 * classifier.
	 */
	public void generateModel() {
		naiveBayes = new NaiveBayes();
		try {
			naiveBayes.buildClassifier(data);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Serialize naive Bayes model.
	 *
	 * @param path the path
	 */
	public void saveModel(String path) {
		try {
			weka.core.SerializationHelper.write(path, naiveBayes);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Cross-validate model and
	 * print validation string.
	 */
	public void crossValidate() {
		Evaluation evaluation = null;
		try {
			evaluation = new Evaluation(data);
			evaluation.crossValidateModel(naiveBayes, data, 10, new Random(1));
			System.out.println(evaluation.toSummaryString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}