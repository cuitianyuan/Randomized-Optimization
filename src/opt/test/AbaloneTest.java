package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class AbaloneTest {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 7, hiddenLayer = 5, outputLayer = 1, trainingIterations = 500;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            double testerror = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
//                
//                TestInstance testoutput = instances[j].getLabel(), example = new TestInstance(network.getOutputValues());
//                example.setLabel(new TestInstance(Double.parseDouble(network.getOutputValues().toString())));
//                testerror += measure.value(testoutput, example);

            }

            System.out.println(df.format(error) );
//            System.out.println(df.format(testerror)) ;
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[4177][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("/Users/tcui/ABAGAIL/src/opt/test/abalone.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[7]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 7; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 9 and  11 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0] < 10 ? 0 : 1));
        }

        return instances;
    }
}


//package opt.test;
//
//import dist.*;
//import opt.*;
//import opt.example.*;
//import opt.ga.*;
//import shared.*;
//import func.nn.backprop.*;
//
//import java.util.*;
//import java.io.*;
//import java.text.*;
//
///**
// * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
// * find optimal weights to a neural network that is classifying abalone as having either fewer 
// * or more than 15 rings. 
// *
// * @author Hannah Lau
// * @version 1.0
// */
//public class AbaloneTest {
//    //private static Instance[] instances = initializeInstances();
//
//    private static Instance[] instances = initializeInstances("/Users/tcui/ABAGAIL/src/opt/test/AbaloneTest_1_3341.txt", 3300);
//    private static Instance[] instancesTest = initializeInstances("/Users/tcui/ABAGAIL/src/opt/test/AbaloneTest_3342_4177(upper20%).txt", 835);
//
//    private static int inputLayer = 9, hiddenLayer = 9, outputLayer = 3, trainingIterations = 430;
//    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
//    
//    private static ErrorMeasure measure = new SumOfSquaresError();
//
//    private static DataSet set = new DataSet(instances);
//
//    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
//    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];
//
//    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
//    private static String[] oaNames = {"RHC", "SA", "GA"};
//    private static String results = "";
//
//    private static DecimalFormat df = new DecimalFormat("0.000");
//
//    public static void main(String[] args) {
//        for(int i = 0; i < oa.length; i++) {
//            networks[i] = factory.createClassificationNetwork(
//                new int[] {inputLayer, hiddenLayer, outputLayer});
//            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
//        }
//
//        oa[0] = new RandomizedHillClimbing(nnop[0]);
//        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
//        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);
//
//        for(int i = 0; i < oa.length; i++) {
//            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
//            train(oa[i], networks[i], oaNames[i]); //trainer.train();
//            end = System.nanoTime();
//            trainingTime = end - start;
//            trainingTime /= Math.pow(10,9);
//
//            Instance optimalInstance = oa[i].getOptimal();
//            networks[i].setWeights(optimalInstance.getData());
//
//            double predicted, actual;
//            start = System.nanoTime();
//            for(int j = 0; j < instancesTest.length; j++) {
//                networks[i].setInputValues(instancesTest[j].getData());
//                networks[i].run();
//
//                predicted = instancesTest[j].getLabel().getData().argMax();
//                actual = networks[i].getOutputValues().argMax();
//
//                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
//
//            }
//            end = System.nanoTime();
//            testingTime = end - start;
//            testingTime /= Math.pow(10,9);
//
//            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
//                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
//                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
//                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
//        }
//
//        System.out.println(results);
//    }
//
//    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
//        System.out.println("\nError results for " + oaName + "\n---------------------------");
//
//        for(int i = 0; i < trainingIterations; i++) {
//            oa.train();
//
//            double error = 0;
//            for(int j = 0; j < instances.length; j++) {
//                network.setInputValues(instances[j].getData());
//                network.run();
//
//                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
//                example.setLabel(new Instance(network.getOutputValues() ));
//                error += measure.value(output, example);
//            }
//
//            System.out.println(df.format(error));
//        }
//    }
//
//    private static Instance[] initializeInstances(String fileName, int numLines) {
//
//        double[][][] attributes = new double[numLines][][];
//
//        try {
//            BufferedReader br = new BufferedReader(new FileReader(new File(fileName)));
//
//            for(int i = 0; i < attributes.length; i++) {
//                Scanner scan = new Scanner(br.readLine());
//                scan.useDelimiter(",");
//
//                attributes[i] = new double[2][];
//                attributes[i][0] = new double[9]; // 7 attributes
//                attributes[i][1] = new double[1];
//
//                for(int j = 0; j < 9; j++)
//                    attributes[i][0][j] = Double.parseDouble(scan.next());
//
//                attributes[i][1][0] = Double.parseDouble(scan.next());
//            }
//
//        }
//        catch(Exception e) {
//            e.printStackTrace();
//        }
//
//        Instance[] instances = new Instance[attributes.length];
//
//        for(int i = 0; i < instances.length; i++) {
//            instances[i] = new Instance(attributes[i][0]);
//
//            // Read the classification from the attribute array
//            int c = (int) attributes[i][1][0];
//
//            // Create a double array of length nClasses, all values are initialized to 0
//            double[] classes = new double[3];
//
//            // Set the i'th index to 1.0 (one-hot encoding)
//            classes[c] = 1.0;
//            instances[i].setLabel(new Instance(classes));
//        }
//
//        return instances;
//    }
//}