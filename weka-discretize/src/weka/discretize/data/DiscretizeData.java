package weka.discretize.data;

import java.util.Random;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class DiscretizeData {
	private static Instances data;
	private NaiveBayes naiveBayes;

	/**
	 * Instantiate DiscretizeData class, discretize data loaded from file, generate
	 * and save naive Bayes model and validate model
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		DiscretizeData disc = new DiscretizeData();
		disc.loadDataFile("data.arff");
		disc.discretizeData();
		disc.generateModel();
		disc.saveModel("naiveBayes.model");
		disc.validate();
	}

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
	 * Discretize data loaded from CSV file, set class index for last attribute in
	 * file, replace missing values and discretize data and set options and import
	 * format for data
	 * 
	 * @return
	 * @throws Exception
	 */
	public Instances discretizeData() throws Exception {

		// remove useless values
		RemoveUseless removeUseless = new RemoveUseless();
		removeUseless.setOptions(new String[] { "-M", "99" });
		removeUseless.setInputFormat(data);
		data = Filter.useFilter(data, removeUseless);

		// replace missing values
		ReplaceMissingValues fixMissing = new ReplaceMissingValues();
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);

		// Discretize data
		Discretize discretize = new Discretize();
		discretize.setOptions(new String[] { "-R", "first-last" });
		discretize.setInputFormat(data);
		data = Filter.useFilter(data, discretize);
		
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setOptions(new String[] { "-T", "0.001" });
		AttributeSelection attSelect = new AttributeSelection();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(ranker);
		attSelect.SelectAttributes(data);
		data = attSelect.reduceDimensionality(data);
		System.out.println(data);

		return data;
	}

	/**
	 * Generate naive Bayes classifier
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
	 * Serialize naive Bayes model
	 * 
	 * @param path
	 */
	public void saveModel(String path) {
		try {
			weka.core.SerializationHelper.write(path, naiveBayes);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Cross-validate model and print summary string
	 */
	public void validate() {
		Evaluation eval = null;
		try {
			eval = new Evaluation(data);
			eval.crossValidateModel(naiveBayes, data, 10, new Random(1));
			System.out.println(eval.toSummaryString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}