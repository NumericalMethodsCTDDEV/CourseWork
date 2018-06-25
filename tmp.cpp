#include <iostream>
#include <cmath>

using namespace std;

int PgAlCl = 0;
int PgAlCl2 = 0;
int PgAlCl3 = 0;
int PgH2 = 0;
int PgHCl = 10000;
int PgN2 = 90000;
int PAtm = 100000;

int R = 8314;

struct component {
    string Ind, Comment;
    char Phase;
    double T1, T2, H, f1, f2, f3, f4, f5, f6, f7, mu, sigma, epsil;

    component(string Indq, char Phaseq, double T1q, double T2q, double Hq, double f1q, double f2q, double f3q, double f4q, double f5q, double f6q, double f7q, double muq, double sigmaq, double epsilq, string Commentq)
    : Ind(Indq), Phase(Phaseq), T1(T1q), T2(T2q), H(Hq), f1(f1q), f2(f2q), f3(f3q), f4(f4q), f5(f5q), f6(f6q), f7(f7q), mu(muq), sigma(sigmaq), epsil(epsilq), Comment(Commentq) {
    }

    double x(double T) {
        return T / 10000;
    }

    double Phi(double T) {
        double xc = x(T);
        return f1 + f2 * log(xc) + f3 / (xc * xc) + f4 / xc + f5 * xc + f6 * (xc * xc) + f7 * (xc * xc * xc);
    }

    double G(double T) {
        return H - Phi(T) * T;
    }

    double D(double T);
};

component alcl = component("AlCl", 'g', 298.15, 3000.0, -51032.0, 318.9948, 36.94626, -0.001226431, 1.1881743, 5.638541, -5.066135, 5.219347, 62.4345, 3.58, 932.0, "1,2");
component alcl2 = component("AlCl2", 'g', 298.15, 3000.0, -259000.0, 427.2137, 56.56409, -0.002961273, 1.893842, 12.40072, -22.65441, 21.29898, 97.8875, 5.3, 825.0, "1,2,55");
component alcl3 = component("AlCl3", 'g', 298.15, 3000.0, -584100.0, 511.8114, 81.15042, -0.004834879, 2.752097, 13.40078, -21.28001, 16.92868, 133.3405, 5.13, 472.0, "1,2");
component gacl = component("GaCl", 'g', 298.15, 3000.0, -70553.0, 332.2718, 37.11052, -0.000746187, 1.1606512, 4.891346, -4.467591, 5.506236, 105.173, 3.696, 348.2, "1");
component gacl2 = component("GaCl2", 'g', 298.15, 3000.0, -241238.0, 443.2976, 57.745845, -0.002265112, 1.8755545, 3.66186, -9.356338, 15.88245, 140.626, 4.293, 465.0, "1");
component gacl3 = component("GaCl3", 'g', 298.15, 3000.0, -431573.0, 526.8113, 82.03355, -0.003486473, 2.6855923, 8.278878, -14.5678, 12.8899, 176.080, 5.034, 548.24, "1");
component nh3 = component("NH3", 'g', 298.15, 3000.0, -45940.0, 231.1183, 20.52222, 0.000716251, 0.7677236, 244.6296, -251.69, 146.6947, 17.031, 3.0, 300.0, "26,1");
component h2 = component("H2", 'g', 298.15, 3000.0, 0.0, 205.5368, 29.50487, 0.000168424, 0.86065612, -14.95312, 78.18955, -82.78981, 2.016, 2.93, 34.1, "4,1");
component hcl = component("HCl", 'g', 298.15, 3000.0, -92310.0, 243.9878, 23.15984, 0.001819985, 0.6147384, 51.16604, -36.89502, 9.174252, 36.461, 2.737, 167.1, "1");
component n2 = component("N2", 'g', 298.15, 3000.0, 0.0, 242.8156, 21.47467, 0.001748786, 0.5910039, 81.08497, -103.6265, 71.30775, 28.0135, 3.798, 71.4, "1");

component al = component("Al", 's', 298.15, 933.61, 0.0, 172.8289, 50.51806, -0.00411847, 1.476107, -458.1279, 2105.75, -4168.337, 26.9815, 0.0, 0.0, "1");
component ga = component("Ga", 'l', 302.92, 2000.0, 0.0, 125.9597, 26.03107, 0.001178297, 0.13976, -0.5698425, 0.04723008, 7.212525, 69.723, 0.0, 0.0, "1,100");
component aln = component("AlN", 's', 298.15, 3000.0, -319000.0, 123.1132, 44.98092, -0.00734504, 1.86107, 31.39626, -49.92139, 81.22038, 40.988, 0.0, 0.0, "2");
component gan = component("GaN", 's', 298.15, 2000.0, -114000.0, 160.2647, 52.86351, -0.00799055, 2.113389, 1.313428, -2.441129, 1.945731, 83.730, 0.0, 0.0, "2");

double component::D(double T) {
    double sigma_i_n2 = (sigma + n2.sigma) / 2;
    double eps_i_n2 = pow(epsil * n2.epsil, 0.5);
    double mu_i_n2 = 2 * mu * n2.mu / (mu + n2.mu);
    double omega = 1.074 + pow(T / eps_i_n2, -0.1604);
    return 2.628 * 0.01 * pow(T, 1.5) / (PAtm * sigma_i_n2 * omega * (T / eps_i_n2) * pow(mu_i_n2, 0.5));
}


// double calcG(double D, double Pg, double Pe, int T) {
//     return
// }

int main() {

}
