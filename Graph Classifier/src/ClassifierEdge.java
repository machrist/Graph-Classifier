import org.jgrapht.graph.DefaultWeightedEdge;

/**
 * Overrides the DefaultWieghtedEdge class to make the methods getSource() and 
 * getTarget() accessible.
 * 
 * @author mchristopher
 *
 */
public class ClassifierEdge extends DefaultWeightedEdge{
	
	public ClassifierNode getSourceNode(){
		return (ClassifierNode) super.getSource();
	}
	
	public ClassifierNode getTargetNode(){
		return (ClassifierNode) super.getTarget();
	}
	
}
