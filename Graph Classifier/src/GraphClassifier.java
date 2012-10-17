import java.io.File;
import java.util.Random;

import org.jgrapht.graph.DefaultDirectedWeightedGraph;
import org.jgrapht.graph.DefaultWeightedEdge;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

/**
 * 
 * @author mchristopher
 *
 */
public class GraphClassifier implements Classifier {
	
	/** Number of classifiers in the model */
	int size;
	
	/** Value in (0,1] indicating size of data set used for each weak classifier */
	double p = 0.10;
	
	/** Fully qualified class name of the classifiers used to build ensemble model */
	String classfierName;
	
	/** Arguments used to create classifiers. Passed to weka method Classifier.forName() */
	String classArgs[];
	
	/** Training data set */
	Instances trainData;
	
	/** Set of weak classifiers to search through*/
	DefaultDirectedWeightedGraph<ClassifierNode, DefaultWeightedEdge> graph;
	
	public GraphClassifier(int n, String classifier, String args[]){
		this.size = n;
		this.classfierName = classifier;
		this.classArgs = args;
	}
	
	/**
	 * Build the ensemble classifier model using the given training data.
	 * 
	 * @param data
	 *   The set of data on which to train
	 */
	public void buildClassifier(Instances data) throws Exception{
		
		this.trainData = data;
		int n = trainData.numInstances();
		Resample sampler = new Resample();
		sampler.setInputFormat(trainData);
		sampler.setSampleSizePercent(100.0*p);
		
		graph = new DefaultDirectedWeightedGraph<ClassifierNode, DefaultWeightedEdge>(DefaultWeightedEdge.class);
		
		ClassifierNode src = new ClassifierNode("s");
		ClassifierNode sink = new ClassifierNode("t");
		
		graph.addVertex(src);
		graph.addVertex(sink);
		
		for(int i = 0; i < this.size; ++i){
			
			try {
				
				//Build weak classifier
				ClassifierNode c = new ClassifierNode(String.format("%02d", i));
				c.setClassifier(AbstractClassifier.forName(this.classfierName, this.classArgs));
				
				Instances curdata = Filter.useFilter(trainData, sampler);
				System.out.println(curdata.get(i));
				
				c.buildModel(curdata);
				c.evaluateOnData(trainData);
				
				//Add to graph representation
				graph.addVertex(c);
				
				graph.addEdge(src, c);
				graph.setEdgeWeight(graph.getEdge(src, c), 1.0 - c.getWeight());
				
				graph.addEdge(c, sink);
				graph.setEdgeWeight(graph.getEdge(c, sink), 1.0 - c.getWeight());
				
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println("GraphClassifier.buildClassifier: Couldn't make classifier " + this.classfierName);
			}
		}
		
	}
	
	protected void findPath(){
		
	}
	
	/**
	 * 
	 * @param data
	 * @param n
	 * @param r
	 * @return
	 */
	Instances sampleWithReplacement(Instances data, int n, Random r){
		
		Instances sample = new Instances(data); 
		
		if(r == null){
			r = new Random();
		}
		
		for(int i = 0; i < n; ++i){
			int idx = r.nextInt(data.numInstances());
			sample.add(data.get(idx));
		}
		
		return sample;
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}


	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
	
	//Getters/setters
	
	public void setClassifier(String c){
		this.classfierName = c;
	}
	
	public void setProportion(double p){
		this.p = p;
	}
	
	public double getProportion(){
		return this.p;
	}
	
	public static void main(String args[]){
		
		int n = 10;
		Instances data = null;
		
		CSVLoader csv = new CSVLoader();
		try {
			csv.setFile(new File(args[0]));
			data = csv.getDataSet();
			data.setClassIndex(data.numAttributes() - 1);
			
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		GraphClassifier gc = new GraphClassifier(n, "weka.classifiers.functions.Logistic", null);
		try {
			gc.buildClassifier(data);
			
			System.out.println(gc.graph);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

}
