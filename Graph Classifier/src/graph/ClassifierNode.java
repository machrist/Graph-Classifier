package graph;
import java.io.Serializable;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.Prediction;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Creates a graph node wrapper around a weka Classifier.
 * 
 * @author mchristopher
 */
public class ClassifierNode implements Serializable{

	/** Unique ID for this node */
	String id;
	
	/** Classifier represented by this node */
	Classifier classifier;
	
	/** Weight that should be associated with this node */
	double weight;
	
	/**
	 * Creates a node with the given id.
	 * 
	 * @param id
	 *   Identifier to assign to this node. Should be unique.
	 */
	public ClassifierNode(String id){
		this.id = id;
	}
	
	/**
	 * Builds the classifier model using the given training data.
	 * 
	 * @param data
	 *   Training data used to build the classifier model
	 * @return
	 *   True if the classifierwas built successfully, false otherwise
	 */
	public boolean buildModel(Instances data){
		boolean result = true;
		try {
			classifier.buildClassifier(data);
		} catch (Exception e) {
			e.printStackTrace();
			result = false;
		}
		return result;
	}
	
	/**
	 * Applies the classifier of this node to a data point.
	 * 
	 * @param datum
	 *   The data point to classify
	 * @return
	 *   Array indicating the estimated probabilities of the data point belonging to each class
	 * @throws Exception
	 */
	public double[] distributionForInstance(Instance datum) throws Exception{
		return this.classifier.distributionForInstance(datum);
	}
	
	/**
	 * Applies the classifier of this node to a data point.
	 * 
	 * @param datum
	 *   The data point to classify
	 * @return
	 *   Index of the estimated class of the data point 
	 * @throws Exception
	 */
	public double classifyInstance(Instance datum) throws Exception{
		return this.classifier.classifyInstance(datum);
	}
	
	/**
	 * Evaluates the classifier using the given testing data.
	 * 
	 * This method sets the value of weight for this node based on the accuracy
	 * on the testing data. The method getWeight() should be used to get the 
	 * resulting weight value.
	 * 
	 * @param data
	 *   Testing data used to evaluate the classifier
	 * @return
	 *   True if the classifier could be evaluated on this data, false otherwise
	 */
	public boolean evaluateOnData(Instances data){
		
		boolean result = true;
		EvaluationUtils eval = new EvaluationUtils();
		
		try {
			double right = 0, total = 0;
			
			FastVector<Prediction> results = (FastVector<Prediction>) eval.getTestPredictions(classifier, data);
			
			for(int i = 0; i < results.size(); ++i){
				Prediction p = results.get(i);
				if(p.predicted() == p.actual()){
					right = right + 1;
				}
				total = total + 1;
			}
			
			weight = right/total;
			
		} catch (Exception e) {
			result = false;
		}
		
		return result;
	}
	
	/**
	 * Gets the weight (proportional to the accuracy) that should be applied to 
	 * this node. The accuracy used to compute the weight is determined during 
	 * the evaluateOnData() method, so this method should only be called after 
	 * that method.
	 * 
	 * @return 
	 *   A weight value that should be associated with this node.
	 */
	public double getWeight(){
		return weight;
	}
	
	/**
	 * Determines if this node is equal to another. Determination is based only
	 * on node id.
	 * 
	 * @return
	 *   True if o is a ClassifierNode and o.id == this.id, false otherwise
	 */
	public boolean equals(Object o){
		if(o instanceof ClassifierNode){
			return this.id.equals(((ClassifierNode) o).id);
		}
		return false;
	}
	
	/**
	 * Overridden to ensure equals/hashcode methods are consistent.
	 */
	public int hashCode(){
		return id.hashCode();
	}
	
	//Basic getters/setters
	
	public void setClassifier(Classifier c){
		this.classifier = c;
	}
	
	public Classifier getClassifier(){
		return this.classifier;
	}
	
	public void setID(String id){
		this.id = id;
	}
	
	public String getID(){
		return this.id;
	}
	
	public String toString(){
		return this.id + " (" + this.weight + ")";
	}
}
