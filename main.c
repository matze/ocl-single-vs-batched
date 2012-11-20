/* main.c as part of mgpu
 *
 * Copyright (C) 2011-2012 Matthias Vogelgesang <matthias.vogelgesang@gmail.com>
 *
 * mgpu is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * mgpu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Labyrinth; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 */

#include <CL/cl.h>
#include <glib.h>
#include <stdio.h>

#include "ocl.h"

typedef struct {
    gint num_images;
    guint num_runs;
    guint width;
    guint height;
} Settings;

typedef struct {
    Settings    *settings;
    guint        num_images;
    gsize        image_size;
    cl_kernel   *kernels;
    opencl_desc *ocl;
} Benchmark;

typedef void (*BenchmarkFunc) (Benchmark *benchmark, cl_mem_flags flags);

static void
generate_2d_data (gfloat *buffer,
                  guint start,
                  guint width,
                  guint height)
{
    for (guint i = start; i < start + width*height; i++)
        buffer[i] = (gfloat) i;
}

static Benchmark *
setup_benchmark(opencl_desc *ocl, Settings *settings)
{
    Benchmark *b;
    cl_program program;
    cl_int errcode = CL_SUCCESS;

    program = ocl_get_program(ocl, "kernels.cl", "");

    if (program == NULL) {
        g_warning ("Could not open kernels.cl");
        ocl_free (ocl);
        return NULL;
    }

    b = (Benchmark *) g_malloc0(sizeof(Benchmark));
    b->ocl = ocl;
    b->settings = settings;

    /* Create kernel for each device */
    b->kernels = g_malloc0(ocl->num_devices * sizeof(cl_kernel));

    for (int i = 0; i < ocl->num_devices; i++) {
        b->kernels[i] = clCreateKernel (program, "take_neg_log", &errcode);
        CHECK_ERROR(errcode);
    }

    b->num_images = b->settings->num_images < 0 ? ocl->num_devices * 16 : b->settings->num_images;
    b->image_size = b->settings->width * b->settings->height * sizeof(gfloat);

    g_print("# Computing <nlm> for %i images of size %ix%i\n", b->num_images, b->settings->width, b->settings->height);

    return b;
}

static void
teardown_benchmark (Benchmark *b)
{
    g_free (b);
}

static void
measure_benchmark (const gchar *prefix, BenchmarkFunc func, Benchmark *benchmark, cl_mem_flags flags)
{
    gdouble time;
    GTimer *timer;
    
    timer = g_timer_new();
    func (benchmark, flags);
    g_timer_stop (timer);
    time = g_timer_elapsed (timer, NULL);
    g_timer_destroy(timer);

    g_print("# %s: total = %fs\n", prefix, time);
}

static void
execute_single_buffer (Benchmark *b, cl_mem_flags flags)
{
    gfloat  *host_mem;
    cl_mem   dev_mem;
    cl_int   err;
    cl_event event;

    size_t global_work_size[] = { b->settings->width, b->settings->height };

    host_mem = g_malloc0 (b->image_size);
    dev_mem = clCreateBuffer (b->ocl->context,
                              flags | CL_MEM_READ_WRITE,
                              b->image_size,
                              host_mem,
                              &err);

    for (guint i = 0; i < b->settings->num_runs; i++) {
        generate_2d_data (host_mem, 0, b->settings->width, b->settings->height);

        if (!(flags & CL_MEM_USE_HOST_PTR)) {
            clEnqueueWriteBuffer (b->ocl->cmd_queues[0],
                                  dev_mem, CL_TRUE,
                                  0, b->image_size,
                                  host_mem,
                                  0, NULL, NULL);
        }

        clEnqueueNDRangeKernel (b->ocl->cmd_queues[0],
                                b->kernels[0],
                                2, NULL, global_work_size, NULL,
                                0, NULL, &event);
        clWaitForEvents (1, &event);
        clReleaseEvent (event);
    }

    clReleaseMemObject (dev_mem);
    g_free (host_mem);
}

static void
execute_batched_buffer (Benchmark *b, cl_mem_flags flags)
{
    gfloat  *host_mem;
    cl_mem   dev_mem;
    cl_int   err;
    cl_event event;
    guint    batch_size;
    gsize    buffer_size;
    size_t   global_work_size[2];

    batch_size = 8;
    buffer_size = b->image_size * batch_size;

    host_mem = g_malloc0 (buffer_size);
    dev_mem = clCreateBuffer (b->ocl->context,
                              flags | CL_MEM_READ_WRITE,
                              buffer_size,
                              host_mem,
                              &err);

    global_work_size[0] = b->settings->width;
    global_work_size[1] = batch_size * b->settings->height;

    for (guint i = 0; i < b->settings->num_runs / batch_size; i++) {
        for (guint j = 0; j < batch_size; j++) {
            generate_2d_data (host_mem,
                              j * b->settings->width * b->settings->height,
                              b->settings->width,
                              b->settings->height);
        }

        if (!(flags & CL_MEM_USE_HOST_PTR)) {
            clEnqueueWriteBuffer (b->ocl->cmd_queues[0],
                                  dev_mem, CL_TRUE,
                                  0, b->image_size,
                                  host_mem,
                                  0, NULL, NULL);
        }

        clEnqueueNDRangeKernel (b->ocl->cmd_queues[0],
                                b->kernels[0],
                                2, NULL, global_work_size, NULL,
                                0, NULL, &event);
        clWaitForEvents (1, &event);
        clReleaseEvent (event);
    }

    clReleaseMemObject (dev_mem);
    g_free (host_mem);
    
}

int
main(int argc, char *argv[])
{
    static Settings settings = {
        .num_runs = 32,
        .num_images = -1,
        .width = 1024,
        .height = 1024,
    };

    static GOptionEntry entries[] = {
        { "num-images", 'n', 0, G_OPTION_ARG_INT, &settings.num_images, "Number of images", "N" },
        { "num-runs", 'r', 0, G_OPTION_ARG_INT, &settings.num_runs, "Number of runs", "N" },
        { "width", 'w', 0, G_OPTION_ARG_INT, &settings.width, "Width of imags", "W" },
        { "height", 'h', 0, G_OPTION_ARG_INT, &settings.height, "Height of images", "H" },
        { NULL }
    };

    GOptionContext *context;
    opencl_desc *ocl;
    Benchmark *benchmark;
    GError *error = NULL;

    context = g_option_context_new (" - test multi GPU performance");
    g_option_context_add_main_entries (context, entries, NULL);

    if (!g_option_context_parse (context, &argc, &argv, &error)) {
        g_print ("Option parsing failed: %s\n", error->message);
        return 1;
    }

    g_print("## %s@%s\n", g_get_user_name(), g_get_host_name());

    g_thread_init (NULL);

    ocl = ocl_new (FALSE);
    benchmark = setup_benchmark (ocl, &settings);

    measure_benchmark ("Single/Copy", execute_single_buffer, benchmark, 0);
    measure_benchmark ("Single/Pinned", execute_single_buffer, benchmark, CL_MEM_USE_HOST_PTR);
    measure_benchmark ("Batched/Copy", execute_batched_buffer, benchmark, 0);
    measure_benchmark ("Batched/Pinned", execute_batched_buffer, benchmark, CL_MEM_USE_HOST_PTR);

    teardown_benchmark (benchmark);
    ocl_free (ocl);
    return 0;
}
