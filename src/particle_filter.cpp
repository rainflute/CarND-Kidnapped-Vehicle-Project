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

// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = 100;
    
    // define normal distributions for sensor noise
    normal_distribution<double> N_x(x, std[0]);
    normal_distribution<double> N_y(y, std[1]);
    normal_distribution<double> N_theta(theta, std[2]);
    
    // init particles
    for (int i = 0; i < num_particles; i++) {
        Particle particle;
        particle.id = i;
        particle.x = N_x(gen);
        particle.y = N_y(gen);
        particle.theta = N_theta(gen);
        particle.weight = 1.0;
        
        particles.push_back(particle);
        weights.push_back(particle.weight);
    }
    // set to initialized
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    
    // define normal distributions for sensor noise
    
    
    for (int i = 0; i < num_particles; i++) {
        double new_x;
        double new_y;
        double new_theta;
        
        // calculate new state
        if (fabs(yaw_rate) < 0.0001) {
            //when yaw_rate is very small or zero
            new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
            new_theta = particles[i].theta;
        }
        else {
            new_x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            new_y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            new_theta = particles[i].theta + yaw_rate * delta_t;
        }
        //define normal distribution
        normal_distribution<double> N_x(new_x, std_pos[0]);
        normal_distribution<double> N_y(new_y, std_pos[1]);
        normal_distribution<double> N_theta(new_theta, std_pos[2]);
        // add noise
        particles[i].x = N_x(gen);
        particles[i].y = N_y(gen);
        particles[i].theta = N_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    
    for (int i = 0; i < observations.size(); i++) {
        
        // grab current observation
        LandmarkObs observation = observations[i];
        
        // init minimum distance to negative number
        double min_dist = -1.0;
        
        // init id of landmark to negative number
        int lm_id = -1;
        
        for (unsigned int j = 0; j < predicted.size(); j++) {
            //current prediction
            LandmarkObs prediction = predicted[j];
            
            //current distance between observation and prediction
            double current_dist = dist(observation.x, observation.y, prediction.x, prediction.y);
            
            // find the predicted landmark nearest the current observed landmark
            if (current_dist < min_dist || min_dist < 0) {
                min_dist = current_dist;
                lm_id = prediction.id;
            }
        }
        
        // set the observation's id to the nearest predicted landmark's id
        observations[i].id = lm_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html
    
    // for each particle...
    for (int i = 0; i < num_particles; i++) {
        // transform observation from car coordinates to map coordinates
        vector<LandmarkObs> transformed_observations;
        for (int j = 0; j < observations.size(); j++) {
            double t_x = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
            double t_y = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
            transformed_observations.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
        }
        
        // landmarks thats in the range of the sensors
        vector<LandmarkObs> lm_in_range;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            // get id and x,y coordinates
            float lm_x = map_landmarks.landmark_list[j].x_f;
            float lm_y = map_landmarks.landmark_list[j].y_f;
            int lm_id = map_landmarks.landmark_list[j].id_i;
            // only associate those in range.
            // this will increase the performance
            if (fabs(lm_x - particles[i].x) <= sensor_range && fabs(lm_y - particles[i].y) <= sensor_range) {
                // add prediction to vector
                lm_in_range.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
            }
        }
        
        // perform dataAssociation for the predictions and transformed observations on current particle
        dataAssociation(lm_in_range, transformed_observations);
        
        double new_weight = 1.0;
        for (int j = 0; j < transformed_observations.size(); j++) {
            double observed_x = transformed_observations[j].x;
            double observed_y = transformed_observations[j].y;
            double associated_lm_x,associated_lm_y;
        
            double s_x = std_landmark[0];
            double s_y = std_landmark[1];
            int association = transformed_observations[j].id;
        
            // get the x,y coordinates of the prediction associated with the current observation
            for (int k = 0; k < lm_in_range.size(); k++) {
                if (lm_in_range[k].id == association) {
                    associated_lm_x = lm_in_range[k].x;
                    associated_lm_y = lm_in_range[k].y;
                }
            }
            
            new_weight *= ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(associated_lm_x-observed_x,2)/(2*pow(s_x, 2)) + (pow(associated_lm_y-observed_y,2)/(2*pow(s_y, 2))) ) );
        }
        //set new weight in particle
        particles[i].weight = new_weight;
        weights[i] = new_weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle> resampled_particles;
    vector<double> resampled_weights;
    discrete_distribution<int> discrete_dist(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; i++) {
        int idx = discrete_dist(gen);
        Particle p = particles[idx];
        resampled_particles.push_back(p);
        resampled_weights.push_back(p.weight);
    }
    particles = resampled_particles;
    weights = resampled_weights;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
