package graph;

import java.io.File;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import org.jgrapht.graph.DefaultDirectedWeightedGraph;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class LayeredGraphClassifier extends GraphClassifier {

	
	/** The number of layers in the graph, this is also the number of classifiers the ensemble method must use. **/
	private int numLayers = 3;
	
	/** Number of classifers in each layer */
	private int numClassifiersPerLayer = 2;
	
	public LayeredGraphClassifier(){
		this(5,5, "weka.classifiers.functions.Logistic", null);
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
	public LayeredGraphClassifier(int numLayers, int numClassifiersPerLayer, String classifier, String args[]){
		this.numLayers = numLayers;
		this.numClassifiersPerLayer = numClassifiersPerLayer;
		this.classfierName = classifier;
		this.classArgs = args;
		System.out.println("NumLayers: "+numLayers+" numClassifiersPerLayer: "+numClassifiersPerLayer);
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
		
		System.out.println("NumLayers: "+numLayers+" numClassifiersPerLayer: "+numClassifiersPerLayer);
		
		System.out.println("Size of original dataset: "+trainData.size());
		
		for(int i = 0; i < this.numClassifiersPerLayer*this.numLayers; ++i){

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
			System.out.println(curdata.get(0));
			

			System.out.println("First instance: "+curdata.size());
			
			c.buildModel(curdata);
			c.evaluateOnData(trainData);
			System.out.println(c.getWeight());
			
			
			//Add to graph representation
			graph.addVertex(c);

			//If it's in the first layer
			if( i/numClassifiersPerLayer == 0 ) {
				graph.addEdge(src, c);
				graph.setEdgeWeight(graph.getEdge(src, c), 1.0 - c.getWeight());
			}
			
			//if it's in the last layer
			if( i/numClassifiersPerLayer == numLayers - 1 ) {
				graph.addEdge(c, sink);
				graph.setEdgeWeight(graph.getEdge(c, sink), b);
			}
		}
		
		makeEdges();
		findShortestPath();
	}
	
	/**
	 * Creates edges for layered classifier. Each node in a layer is connected
	 * every node in the next layer. This forces the graph algorithm to use one
	 * classifier from each layer, and removes the problem of cycles.
	 * 
	 * @throws Exception 
	 */
	protected void makeEdges() throws Exception{
		
		Vector<ClassifierEdge> curEdge = new Vector<ClassifierEdge>(1);
		curEdge.add(null);
		
		Vector<ClassifierNode> vertices = new Vector<ClassifierNode>();
		vertices.addAll(this.graph.vertexSet());
		vertices.remove(this.src);
		vertices.remove(this.sink);
		
		for(int i = 0; i < numLayers - 1; i++) {
			
			for(int j = 0; j < numClassifiersPerLayer; j++) {
				
				int classiferId = i*numClassifiersPerLayer + j;
				
				ClassifierNode ci = vertices.get( classiferId );
				
				
				for(int k = 0; k < numClassifiersPerLayer; k++ ) {
					
					int adjClassifier = (i+1)*numClassifiersPerLayer + k;
					
					ClassifierNode cj = vertices.get( adjClassifier );
					
					this.graph.addEdge(ci, cj);
					
					ClassifierEdge cicj = graph.getEdge(ci, cj);
					curEdge.set(0, cicj);
					
					LayeredPathClassifier pc = new LayeredPathClassifier(curEdge);
					pc.buildClassifier(trainData);
					double acc = pc.evaluateOnData(trainData);
					
					System.out.println("Edge: " + ci + " -> " + cj + ": acc = " + acc + ", w = " + ((1.0 - acc) - (1.0 - ci.getWeight())));
					
					this.graph.setEdgeWeight(graph.getEdge(ci, cj), (1.0 - acc) - (1.0 - ci.getWeight()));
					
				}
				
				
			}
			
		}
		
		/*
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
				
//				System.out.println("Edge: " + ci + " -> " + cj + ": acc = " + acc + ", w = " + ((1.0 - acc) - (1.0 - ci.getWeight())));
//				System.out.println("Edge: " + cj + " -> " + ci + ": acc = " + acc + ", w = " + ((1.0 - acc) - (1.0 - cj.getWeight())));
				
				this.graph.setEdgeWeight(graph.getEdge(ci, cj), (1.0 - acc) - (1.0 - ci.getWeight()));
				this.graph.setEdgeWeight(graph.getEdge(cj, ci), (1.0 - acc) - (1.0 - cj.getWeight()));
			}
		}
		
		*/
		
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
		System.out.println(args[0]);
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
		
		LayeredGraphClassifier gc = new LayeredGraphClassifier(3,2, "weka.classifiers.functions.Logistic", null);
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
	
	
	private class LayeredPathClassifier extends PathClassifier {

		public LayeredPathClassifier(List<ClassifierEdge> edges) {
			super(edges);
			// TODO Auto-generated constructor stub
		}
		
		protected double sumOverPath(Instance datum) throws Exception{
			double sum = 0.0;
			
			for(int i = 0; i < this.edges.size(); ++i){
				ClassifierNode ci = this.edges.get(i).getSourceNode();
				ClassifierNode cj = this.edges.get(i).getTargetNode();
				if(!ci.getID().equalsIgnoreCase("s"))
					sum += ci.getWeight()*ci.distributionForInstance(datum)[0];
				if(!ci.getID().equalsIgnoreCase("t"))
					sum += cj.getWeight()*cj.distributionForInstance(datum)[0];
				
			}
			
			return sum;
		}
	
	}
	

}
