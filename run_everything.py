import generate_subimages
import create_targets
import build
import preprocessing

if __name__ == "__main__":
    print('Running create_targets')
    create_targets.run()

    print('Running preprocessing')
    preprocessing.run()

    print('Running generate_subimages')
    generate_subimages.run()

    print('Running build')
    build.run()
