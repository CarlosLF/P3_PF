/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    if (is_initialized) {
        return;
    }
    
    num_particles = 100;
    
    //  Standard deviations
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    
    // Normal distributions
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
    
    particles.resize(num_particles);
    
    // Generate particles with normal distribution with mean on GPS values.
    for (int i = 0; i < num_particles; i++) {
        
        particles[i].id = i;
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].weight = 1.0;
        
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    //  standard deviations
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];
    
    // normal distributions
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);
    
    double delta_theta=0;
    
    for (int i = 0; i < num_particles; i++) {
        
        double theta = particles[i].theta;
        
        if ( fabs(yaw_rate) < EPS ) { ///motion model with zero yaw_rate
            particles[i].x += velocity * delta_t * cos( theta );
            particles[i].y += velocity * delta_t * sin( theta );
            
        } else {//motion model
            delta_theta= yaw_rate * delta_t;
            
            particles[i].x += velocity / yaw_rate * ( sin( theta + delta_theta ) - sin( theta ) );
            particles[i].y += velocity / yaw_rate * ( cos( theta ) - cos( theta + delta_theta ) );
            particles[i].theta += delta_theta;
        }
        
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    int n_observations = observations.size();
    int n_predictions = predicted.size();
    double min_dis, x_dis, y_dis, dis;
    
    for (unsigned int i = 0; i < n_observations; i++) {
        
        min_dis = numeric_limits<double>::max();
        
        int id = -1;
        
        for (unsigned j = 0; j < n_predictions; j++ ) {
            
            x_dis = observations[i].x - predicted[j].x;
            y_dis = observations[i].y - predicted[j].y;
            
            dis = x_dis * x_dis + y_dis * y_dis;
            
            if ( dis < min_dis ) {
                min_dis = dis;
                id = predicted[j].id;
            }
        }
        
        observations[i].id = id;
    }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    double stdLandmarkRange = std_landmark[0];
    double stdLandmarkBearing = std_landmark[1];
    double x,y,theta;
    double landmarkX, landmarkY, dX, dY, sensor_range_2;
    int id;
    double xp, yp;
    
    double observationX, observationY ;
    int landmarkId;
    unsigned int nLandmarks;
    double weight;
    
    for (int i = 0; i < num_particles; i++) {
        
        x = particles[i].x;
        y = particles[i].y;
        theta = particles[i].theta;
        
        sensor_range_2 = sensor_range * sensor_range;
        vector<LandmarkObs> inRangeLandmarks;
        for( int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            landmarkX = map_landmarks.landmark_list[j].x_f;
            landmarkY = map_landmarks.landmark_list[j].y_f;
            id = map_landmarks.landmark_list[j].id_i;
            dX = x - landmarkX;
            dY = y - landmarkY;
            if ( dX*dX + dY*dY <= sensor_range_2 ) {
                inRangeLandmarks.push_back(LandmarkObs{ id, landmarkX, landmarkY });
            }
        }
        
        // 2D Transformation of observation coordinates, from vehicles coordinates to map coordinates
        vector<LandmarkObs> mappedObservations;
        for( int j = 0; j < observations.size(); j++) {
            xp = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
            yp = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
            mappedObservations.push_back(LandmarkObs{ observations[j].id, xp, yp });
        }
        
        // Association of Observations to landmarks
        dataAssociation(inRangeLandmarks, mappedObservations);
        
        // Update particle's weights
        particles[i].weight = 1.0;
        
       for( int j = 0; j < mappedObservations.size(); j++) {
            observationX = mappedObservations[j].x;
            observationY = mappedObservations[j].y;
            
           landmarkId = mappedObservations[j].id;
            
           landmarkX=landmarkY=0;
           
           nLandmarks = inRangeLandmarks.size();
           
           for (int k=0; i < nLandmarks; k++){
               if ( inRangeLandmarks[k].id == landmarkId) {
                   landmarkX = inRangeLandmarks[k].x;
                   landmarkY = inRangeLandmarks[k].y;
                   break;
               }
           }
            
            // Calculating weight.
            dX = observationX - landmarkX;
            dY = observationY - landmarkY;
            
            weight = ( 1/(2*M_PI*stdLandmarkRange*stdLandmarkBearing)) * exp( -( dX*dX/(2*stdLandmarkRange*stdLandmarkRange) + (dY*dY/(2*stdLandmarkBearing*stdLandmarkBearing)) ) );
            if (weight == 0) {
                particles[i].weight *= EPS;
            } else {
                particles[i].weight *= weight;
            }
        }
    }
    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // Compute weight vector w, and max weight (mw)
    vector<double> w;
    double mw = numeric_limits<double>::min();
    for(int i = 0; i < num_particles; i++) {
        w.push_back(particles[i].weight);
        if ( particles[i].weight > mw ) {
            mw = particles[i].weight;
        }
    }
    
    // Distributions
    uniform_real_distribution<double> distDouble(0.0, mw);
    uniform_int_distribution<int> distInt(0, num_particles - 1);
    
    double beta = 0.0;
    int index = distInt(gen);

    // 20. Resampling wheel
    vector<Particle> resampledParticles;
    for(int i = 0; i < num_particles; i++) {
        beta += distDouble(gen) * 2.0;
        while( beta > w[index]) {
            beta -= w[index];
            index = (index + 1) % num_particles;
        }
        resampledParticles.push_back(particles[index]);
    }
    
    particles = resampledParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    
    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
