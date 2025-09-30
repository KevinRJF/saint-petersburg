import java.io.FileNotFoundException;
import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        // Open "cards.csv" and print the header line and the first 5 lines thereafter.
        // If the file does not exist, print "Cannot find cards.csv yet.".
        try (Scanner in = new Scanner(new File("cards.csv"))) {
            if (in.hasNextLine()) {
                System.out.println(in.nextLine()); // Print header line
            }
            for (int i = 0; i < 5 && in.hasNextLine(); i++) {
                System.out.println(in.nextLine()); // Print next 5 lines
            }
        } catch (FileNotFoundException e) {
            System.out.println("Cannot find cards.csv yet.");
        }   
    }
}