import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Classifier that generates a classification for a data point by applying a logistic
 * regression to the weighted sum of classifiers along the path.
 * 
 * @author mchristopher
 *
 */
public class PathClassifier extends AbstractClassifier{

	/** Set of edges defining the path */
	List<ClassifierEdge> edges;
	
//	/** Threshold value separting the two classes */
//	double t = 0;
//	
//	/** Minimum value seen on training data */
//	double min;
//	
//	/** Maximum value seen on training data */
//	double max;
	
	/** Logistic regression fit to weighted path sum values */
	Logistic logistic;
	
	public PathClassifier(List<ClassifierEdge> edges){
		this.setPath(edges);
	}
	
	protected double sumOverPath(Instance datum) throws Exception{
		double sum = 0.0;
		
		for(int i = 0; i < this.edges.size(); ++i){
			ClassifierNode c = this.edges.get(i).getSourceNode();
			sum += c.getWeight()*c.distributionForInstance(datum)[0];
			
		}
		
		return sum;
	}
	
	/**
	 * Apply the classifiers on this path to the given data point.
	 * 
	 * @param datum
	 * @return
	 * @throws Exception 
	 */
	public double classifyInstance(Instance datum) throws Exception{
		
		Instance d = new DenseInstance(2);
		d.setValue(0, this.sumOverPath(datum));
		d.setValue(1, datum.classValue());
		
		return logistic.classifyInstance(d);
		
//		return this.sumOverPath(datum) < t ? 0 : 1;
	}
	
	/**
	 * 
	 */
	public double[] distributionForInstance(Instance datum) throws Exception{
		
		Instance d = new DenseInstance(2);
		d.setValue(0, this.sumOverPath(datum));
		d.setValue(1, datum.classValue());
		
		return logistic.distributionForInstance(d);
	}
	
	/**
	 * 
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double evaluateOnData(Instances data) throws Exception{
		
		EvaluationUtils eval = new EvaluationUtils();

		double right = 0, total = 0;

		FastVector<Prediction> results = (FastVector<Prediction>) eval.getTestPredictions(this, data);

		for(int i = 0; i < results.size(); ++i){
			Prediction p = results.get(i);
			if(p.predicted() == p.actual()){
				right = right + 1;
			}
			total = total + 1;
		}

		return right/total;
		
		
//		double right = 0.0, total = 0.0;
//		
//		for(int i = 0; i < data.numInstances(); ++i){
//			
//			Instance datum = data.get(i);
//			
//			double cidx = this.classifyInstance(datum);
//			double tidx = datum.classValue();
//			
//			if(cidx == tidx){
//				right += 1.0;
//			}
//			total += 1.0;
//		}
//		
//		return right/total;
	}

	/**
	 * 
	 */
	public void buildClassifier(Instances data) throws Exception {
		
		FastVector<String> classes = new FastVector<String>(2);
		classes.add("0");
		classes.add("1");
		
		FastVector<Attribute> atts = new FastVector<Attribute>(); 
		atts.add(new Attribute("sums"));
		atts.add(new Attribute("class", classes));
		
		Instances sums = new Instances("sums", atts, data.numInstances());
		sums.setClassIndex(1);
		
		for(int i = 0; i < data.numInstances(); ++i){
			Instance datum = data.get(i);
			double s = this.sumOverPath(datum);
			double c = datum.classValue();
			
			Instance sdatum = new DenseInstance(2);
			
			sdatum.setValue(0, s);
			sdatum.setValue(1, c);
		}
		
		
		logistic = new Logistic();
		logistic.buildClassifier(data);
		
	}
	
	//Getters/Setters
	public void setPath(List<ClassifierEdge> edges){
		this.edges = edges;
	}
	
//	public void setThreshold(double t){
//		this.t = t;
//	}
//	
//	public double getThreshold(){
//		return this.t;
//	}
}
