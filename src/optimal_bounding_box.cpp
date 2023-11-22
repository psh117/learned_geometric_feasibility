#ifndef CGAL_EIGEN3_ENABLED
#define CGAL_EIGEN3_ENABLED
#endif

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/optimal_bounding_box.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Real_timer.h>
#include <fstream>
#include <iostream>

namespace PMP = CGAL::Polygon_mesh_processing;
typedef CGAL::Exact_predicates_inexact_constructions_kernel    K;
typedef K::Point_3                                             Point;
typedef CGAL::Surface_mesh<Point>                              Surface_mesh;
typedef CGAL::Aff_transformation_3<K>                          Aff_transformation_3;
int main(int argc, char** argv)
{
    std::string full_path = std::string(argv[1]);
    std::string file_name = std::string(argv[1]).substr(full_path.find_last_of("/")+1, full_path.find_last_of(".") - full_path.find_last_of("/") - 1);
    std::cout << file_name << std::endl;
    Surface_mesh sm;
    CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(full_path, sm);
    
    CGAL::Real_timer timer;
    timer.start();
    // Compute the extreme points of the mesh, and then a tightly fitted oriented bounding box
    std::array<Point, 8> obb_points;
    Aff_transformation_3 aff_transform;

    CGAL::oriented_bounding_box(sm, aff_transform);

    std::cout << "Elapsed time: " << timer.time() << std::endl;
    std::ofstream ofile("affine_" + file_name+ ".txt");

    for (int i=0; i<4; ++i)
    {
        for (int j=0; j<4; ++j)
        {
            ofile << aff_transform.m(i,j) << " ";
        }
        ofile << std::endl;
    }
    
    // Make a mesh out of the oriented bounding box
    // Surface_mesh obb_sm;
    // CGAL::make_hexahedron(obb_points[0], obb_points[1], obb_points[2], obb_points[3],
    //                         obb_points[4], obb_points[5], obb_points[6], obb_points[7], obb_sm);
    // std::ofstream("obb_"+ file_name +".off") << obb_sm;
    // PMP::triangulate_faces(obb_sm);
    // std::cout << "Volume: " << PMP::volume(obb_sm) << std::endl;

    // std::ofstream("obb_points_" + file_name+ ".txt") << obb_points[0] << std::endl
    //                                                             << obb_points[1] << std::endl
    //                                                             << obb_points[2] << std::endl
    //                                                             << obb_points[3] << std::endl
    //                                                             << obb_points[4] << std::endl
    //                                                             << obb_points[5] << std::endl
    //                                                             << obb_points[6] << std::endl
    //                                                             << obb_points[7] << std::endl;
    return EXIT_SUCCESS;
}