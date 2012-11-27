package graph;
import java.io.File;
import java.io.Serializable;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import org.jgrapht.alg.BellmanFordShortestPath;
import org.jgrapht.graph.DefaultDirectedWeightedGraph;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

/**
 * Graph-based ensemble classifier. Uses shortest-path graph search techniques to search through
 * a set of weak classifiers to determine a stronger ensemble classification model. 
 * 
 * @author mchristopher
 *
 */
public class GraphClassifier implements Classifier, Serializable, OptionHandler {
	
	/** Number of classifiers in the model */
	int size;
	
	/** Value in (0,1] indicating size of data set used for each weak classifier */
	double p = 0.10;
	
	/** Weight connecting each weak classifier to sink. Serves as threshold for graph search. */
	double b = 0.00;
	
	/** Fully qualified class name of the classifiers used to build ensemble model */
	String classfierName;
	
	/** Arguments used to create classifiers. Passed to weka method Classifier.forName() */
	String classArgs[];
	
	/** Capabilities of the classifier, same as for weak classifier */
	Capabilities caps;
	
	/** Training data set */
	Instances trainData;
	
	/** Set of weak classifiers to search through*/
	DefaultDirectedWeightedGraph<ClassifierNode, ClassifierEdge> graph;
	
	/** Starting node in graph */
	ClassifierNode src;
	
	/** Ending node in graph */
	ClassifierNode sink;
	
	/** Path representing the best set of weak classifiers */
	PathClassifier path;
	
	String[] args = new String[0];
	
	public GraphClassifier(){
		this(10, "weka.classifiers.functions.Logistic", null);
	}
	
	/**
	 * Creates a graph-based classifier with a size and node type specified by the
	 * input. The ensemble graph classifier will be determined by searching through
	 * n weaker classifiers each constructed using the Weka method 
	 * Classifier.forName(classifier, args). 
	 * 
	 * @param n
	 *   Number of weak classifiers to create/search through
	 * @param classifier
	 *    Fully qualified name of classifier type to use for weak classifiers
	 * @param args
	 *    Arguments to 
	 */
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
		
		this.caps = null;
		this.trainData = data;
		
		Random rand = new Random();
		
		graph = new DefaultDirectedWeightedGraph<ClassifierNode,ClassifierEdge>(ClassifierEdge.class);
		
		src = new ClassifierNode("s");
		sink = new ClassifierNode("t");
		
		graph.addVertex(src);
		graph.addVertex(sink);
		
		for(int i = 0; i < this.size; ++i){

			//Build weak classifier
			ClassifierNode c = new ClassifierNode(this.getClassifierName(i));
			c.setClassifier(AbstractClassifier.forName(this.classfierName, this.classArgs));

			if(this.caps == null){
				this.caps = c.getClassifier().getCapabilities();
			}
			
			Resample sampler = new Resample();
			sampler.setInputFormat(trainData);
			sampler.setSampleSizePercent(100.0*p);
			sampler.setRandomSeed(rand.nextInt());
			
			Instances curdata = Filter.useFilter(trainData, sampler);
			System.out.println(curdata.get(i));

			c.buildModel(curdata);
			c.evaluateOnData(trainData);

			//Add to graph representation
			graph.addVertex(c);

			graph.addEdge(src, c);
			graph.setEdgeWeight(graph.getEdge(src, c), 1.0 - c.getWeight());

			graph.addEdge(c, sink);
			graph.setEdgeWeight(graph.getEdge(c, sink), b);
		}
		
		makeEdges();
		findShortestPath();
	}
	
	/**
	 * Creates edges connecting all classifier nodes to each other. The weights assigned
	 * to each edge are determined by the marginal increase in error associated with
	 * combining the two classifiers connected by the edge.
	 * 
	 * @throws Exception 
	 */
	protected void makeEdges() throws Exception{
		
		Vector<ClassifierEdge> curEdge = new Vector<ClassifierEdge>(1);
		curEdge.add(null);
		curEdge.add(null);
		
		Vector<ClassifierNode> vertices = new Vector<ClassifierNode>();
		vertices.addAll(this.graph.vertexSet());
		vertices.remove(this.src);
		vertices.remove(this.sink);
		
		for(int i = 0; i < this.size; ++i){
			
			ClassifierNode ci = vertices.get(i);
			
			for(int j = i + 1; j < this.size; ++j){
				ClassifierNode cj = vertices.get(j);
				
				this.graph.addEdge(ci, cj);
				this.graph.addEdge(cj, ci);
				
				ClassifierEdge cicj = graph.getEdge(ci, cj);
				ClassifierEdge cjsink = graph.getEdge(cj, sink);
				curEdge.set(0, cicj);
				curEdge.set(1, cjsink);
				
				PathClassifier pc = new PathClassifier(curEdge);
				pc.buildClassifier(trainData);
				double acc = pc.evaluateOnData(trainData);
				
				System.out.println("Edge: " + ci + " -> " + cj + ": acc = " + acc + ", w = " + ((1.0 - acc) - (1.0 - ci.getWeight())));
				System.out.println("Edge: " + cj + " -> " + ci + ": acc = " + acc + ", w = " + ((1.0 - acc) - (1.0 - cj.getWeight())));
				
				this.graph.setEdgeWeight(graph.getEdge(ci, cj), (1.0 - acc) - (1.0 - ci.getWeight()));
				this.graph.setEdgeWeight(graph.getEdge(cj, ci), (1.0 - acc) - (1.0 - cj.getWeight()));
			}
		}
		
	}
	
	/**
	 * Determines final classification model using shortest path search through the graph 
	 * representation of the ensemble of weak classifiers.
	 * 
	 * @throws Exception 
	 */
	protected void findShortestPath() throws Exception{
		List<ClassifierEdge> edges = BellmanFordShortestPath.findPathBetween(this.graph, this.src, this.sink);
		edges.remove(0);
		edges.remove(edges.size()-1);
		this.path = new PathClassifier(edges);
		this.path.buildClassifier(trainData);
	}
	
	/**
	 * Get string name of classifer specified by an index.
	 * 
	 * @param i
	 *   Index specifying the classifier
	 * @return 
	 */
	public String getClassifierName(int i){
		return String.format("%03d", i);
	}

	
	/**
	 * Classifies a single data point.
	 * 
	 * @param instance
	 *   Data point for which to predict class
	 * @return 
	 *   Index of predicted class value
	 * @throws
	 */
	public double classifyInstance(Instance instance) throws Exception {
		return this.path.classifyInstance(instance);
	}
	
	/**
	 * Gets probabilities of data point belonging to each class.
	 * 
	 * @param instance
	 *   Data point for which to predict class probabilities
	 * @return
	 *   Array of class probability values, in order of class values provided in training data
	 * @throws
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {
		return this.path.distributionForInstance(instance);
	}
	
	/**
	 * Returns the (weka) capabilities of this classifier. Identical to capabilities of 
	 * weak classifiers.
	 */
	public Capabilities getCapabilities() {
		Capabilities caps = null;
		try{
			caps = AbstractClassifier.forName(this.classfierName, this.classArgs).getCapabilities();
		}
		catch(Exception e){
			caps = null;
		}
		
		return caps;
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
	
	public void setB(double b){
		this.b = b;
	}
	
	public double getB(){
		return this.b;
	}
	
	
	
	/**
	 * @return the classfierName
	 */
	public String getClassfierName() {
		return classfierName;
	}

	/**
	 * @param classfierName the classfierName to set
	 */
	public void setClassfierName(String classfierName) {
		this.classfierName = classfierName;
	}

	/**
	 * @return the classArgs
	 */
	public String getClassArgs() {
		
		String args = "";
		
		if(classArgs != null) {
			for(int i = 0; i < classArgs.length; i++) {
				
				if(i > 0)
					args += " ";
					
				args += classArgs[i];
				
			}
		}
		
		return "";
	}

	/**
	 * @param classArgs the classArgs to set
	 */
	public void setClassArgs(String s) {
		if(s.length() > 0) {
			classArgs = s.split(" ");
		} else {
			classArgs = null;
		}
	}

	/**
	 * Testing method.
	 * 
	 * @param args
	 *   args[0] should be path to data file (csv or arff) on which to test.
	 */
	public static void main(String args[]){
		
		int n = 10;
		Instances data = null;
		
		try {
			
			//Check for csv file
			if(args[0].endsWith("csv")){
				CSVLoader csv = new CSVLoader();
				csv.setFile(new File(args[0]));
				data = csv.getDataSet();
			}
			//Not csv, assume arff
			else{
				ArffLoader arff = new ArffLoader();
				arff.setFile(new File(args[0]));
				data = arff.getDataSet();
			}
			
			data.setClassIndex(data.numAttributes() - 1);
			
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		GraphClassifier gc = new GraphClassifier(n, "weka.classifiers.functions.Logistic", null);
		gc.setProportion(0.5);
		gc.setB(0.01);
		try {
			gc.buildClassifier(data);
			
			System.out.println(gc.graph);
			System.out.println(gc.path);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	public Enumeration listOptions() {
		// TODO Auto-generated method stub
		return null;
	}

	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub
		
	}

	public String[] getOptions() {
		// TODO Auto-generated method stub
		return args;
	}

}
