import java.io.File;
import java.util.Random;

import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.instance.Resample;


public class GraphClassifier implements Classifier {
	
	/** Number of classifiers in the model */
	int size;
	
	/** Fully qualified class name of the classifiers used to build ensemble model */
	String classfierName;
	
	/** Arguments used to create classifiers. Passed to weka method Classifier.forName() */
	String classArgs[];
	
	/** Training data set */
	Instances trainData;
	
	public GraphClassifier(int n, String classifier, String args[]){
		
	}
	
	public void buildClassifier(Instances data){
		this.trainData = data;
		Resample sampler = null;
		Random rand = new Random();
		
		SimpleGraph<ClassifierNode, DefaultEdge> model = new SimpleGraph<ClassifierNode, DefaultEdge>(DefaultEdge.class);
		
		ClassifierNode src = new ClassifierNode("s");
		ClassifierNode sink = new ClassifierNode("t");
		
		model.addVertex(src);
		model.addVertex(sink);
		
		for(int i = 0; i < this.size; ++i){
			ClassifierNode c = new ClassifierNode(String.format("%02d", i));
			model.addVertex(c);
			model.addEdge(src, c);
			model.addEdge(c, sink);
		}
		
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
	
	
	public void setClassifier(String c){
		this.classfierName = c;
	}
	
	public static void main(String args[]){
		
		int n = 100;
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
		
		
		GraphClassifier gc = new GraphClassifier(n, "weka.classifiers.trees.DecisionStump", null);
		
	}

}
