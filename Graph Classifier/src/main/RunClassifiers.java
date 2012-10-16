package main;

import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.GUIChooser;


public class RunClassifiers {

	public static final int FOLDS = 10;
	
	/**
	 * These values are taken from the gui
	 * Go to the explorer and run the classifer and then the options are at the very top of the classifier output
	 */
	private static String baggingOpts[] = {"-P","100","-S","1","-num-slots","1","-I","10","-W","weka.classifiers.trees.REPTree","--","-M","2","-V","0.001","-N","3","-S","1","-L","-1"};
	private static String randomForestOpts[] = {"-I","10","-K","0","-S","1"};
	private static String boostingOpts[] = {"-P","100","-S","1","-I","10","-W","weka.classifiers.trees.DecisionStump"};
	
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		DataSource source = new DataSource(args[0]);
		//DataSource source = new DataSource("/Users/krtaylor/Documents/CBCB/Graph-Classifier/Graph Classifier/Datasets/breast-cancer-wisconsin.data.arff");
		
		Instances data = source.getDataSet();

		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		Random rand = new Random(System.currentTimeMillis());   // create seeded number generator
		Instances randData = new Instances(data);   // create copy of original data
		randData.randomize(rand); 

		randData.stratify(FOLDS);
		
		//Create classifiers
		RandomForest rfClassifier = new RandomForest();
		rfClassifier.setOptions(randomForestOpts);
		
		AdaBoostM1 boosting = new AdaBoostM1();
		boosting.setOptions(boostingOpts);
		
		Bagging baggingClassifier = new Bagging();
		baggingClassifier.setOptions(baggingOpts);
		
		//Create evaluators
		Evaluation rfEval = new Evaluation(randData);
		Evaluation boostingEval = new Evaluation(randData);
		Evaluation baggingEval = new Evaluation(randData);
		
		
		
		for (int n = 0; n < FOLDS; n++) {
		   Instances train = randData.trainCV(FOLDS, n);
		   Instances test = randData.testCV(FOLDS, n);
		 
		   
		   //RandomForest
		   Classifier rfCopy = RandomForest.makeCopy(rfClassifier);
		   rfCopy.buildClassifier(train);
		   rfEval.evaluateModel(rfCopy,test);
		   
		   //Boosting
		   Classifier boostingCopy = AdaBoostM1.makeCopy(boosting);
		   boostingCopy.buildClassifier(train);
		   boostingEval.evaluateModel(boostingCopy,test);
		   
		   //Bagging
		   Classifier baggingCopy = baggingClassifier.makeCopy(baggingClassifier);
		   baggingCopy.buildClassifier(train);
		   baggingEval.evaluateModel(baggingCopy,test);
		      
		}
		
		/*
		System.out.println();
	    System.out.println("=== Setup ===");
	    System.out.println();
	    System.out.println(rfEval.toSummaryString("=== " + FOLDS + "-fold Cross-validation ===", false));
	    */
	    
	    System.out.println("Random Forest: "+rfEval.pctCorrect()+" %");
	    System.out.println("Boosting: "+boostingEval.pctCorrect()+" %");
	    System.out.println("Bagging: "+baggingEval.pctCorrect()+" %");
	    
	}

}
