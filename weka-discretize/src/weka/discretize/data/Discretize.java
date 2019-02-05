package weka.discretize.data;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class Discretize {

	public static Instances preProcessData(Instances data) throws Exception{
		
		/* 
		 * Remove useless attributes
		 */
		RemoveUseless removeUseless = new RemoveUseless();
		removeUseless.setOptions(new String[] { "-M", "99" });	// threshold
		removeUseless.setInputFormat(data);
		data = Filter.useFilter(data, removeUseless);

		
		/* 
		 * Remove useless attributes
		 */
		ReplaceMissingValues fixMissing = new ReplaceMissingValues();
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);
		

		/* 
		 * Remove useless attributes
		 */
		Discretize discretizeNumeric = new Discretize();
		((OptionHandler) discretizeNumeric).setOptions(new String[] {
				"-O",
				"-M",  "-1.0", 
				"-B",  "4",  // no of bins
				"-R",  "first-last"}); //range of attributes
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);

		/* 
		 * Select only informative attributes
		 */
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();
		search.setOptions(new String[] { "-T", "0.001" });	// information gain threshold
		AttributeSelection attSelect = new AttributeSelection();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		
		// apply attribute selection
		attSelect.SelectAttributes(data);
		
		// remove the attributes not selected in the last run
		data = attSelect.reduceDimensionality(data);

		return data;
	}
}