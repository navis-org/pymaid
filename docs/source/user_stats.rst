User stats, contributions and logs
----------------------------------
Pymaid has functions that let you look at stats/contributions of individual users as well as your project's history and logs.

There are quite a few, simple examples in the documentation for each function. Examples given here are a bit more elaborate.

In this first example, we will compare two users:

>>> # This assumes you have imported pymaid and setup a CATMAID instance
>>> import matplotlib.pyplot as plt
>>> # Get history for 3 months
>>> hist = pymaid.get_history(start_date=(2017,1,1), end_date=(2017,3,31))
>>> # Create empty plot
>>> fig, ax = plt.subplots( 3, 1, sharex=True )
>>> # Plot cable lenght in top plot
>>> hist.cable.loc[['user_id1','user_id2']].T.plot(ax=ax[0])
>>> ax[0].set_ylabel('cable traced [nm]')
>>> # Plot connector links created in middle plot
>>> hist.connector_links.loc[['user_id1','user_id2']].T.plot(ax=ax[1], legend=False)
>>> ax[1].set_ylabel('links created')
>>> # Plot nodes reviewed in bottom plot
>>> hist.cable.loc[['user_id1','user_id2']].T.plot(ax=ax[2], legend=False)
>>> ax[2].set_ylabel('nodes reviewed')
>>> # Tighten plot
>>> plt.tight_layout()
>>> # Render plot
>>> plt.show()

Reference
=========

.. automodule:: pymaid
    :members: get_user_contributions, get_time_invested, get_history, get_logs, get_contributor_statistics, get_user_list
    :undoc-members:
    :show-inheritance: